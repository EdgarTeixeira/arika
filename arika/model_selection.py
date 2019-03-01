import glob
import itertools
import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections import abc
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Sequence,
                    Union)

import joblib
import tensorflow as tf

_REGISTRY: Dict[str, Any] = {}


def register_loader(name, loader):
    if name in _REGISTRY:
        raise RuntimeError(f'"{name}" is already in registry')
    _REGISTRY[name] = loader


def get_loader(name):
    return _REGISTRY[name]


class Loader(ABC):
    @staticmethod
    @abstractmethod
    def save_model(model, filename):
        pass

    @staticmethod
    @abstractmethod
    def load_model(filename):
        pass


class JoblibLoader(Loader):
    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

    @staticmethod
    def save_model(model, filename):
        joblib.dump(model, filename, compress=3)


class KerasLoader(Loader):
    @staticmethod
    def load_model(filename):
        with open(f'{filename}.arch', 'r') as file:
            model = tf.keras.models.model_from_json(file.read())
        model.load_weights(f'{filename}.h5')
        return model

    @staticmethod
    def save_model(model, filename):
        model_arch = model.to_json()
        with open(f'{filename}.arch', 'w') as file:
            file.write(model_arch)
        model.save_weights(f'{filename}.h5', save_format='h5')


SklearnLoader = JoblibLoader
register_loader('joblib', JoblibLoader)
register_loader('sklearn', SklearnLoader)
register_loader('keras', KerasLoader)


class Template(ABC):
    def __init__(self, loader: str) -> None:
        self.loader = loader

    @abstractmethod
    def build_model(self, hyper_parameters):
        pass

    @abstractmethod
    def fit_model(self, model, x_train, y_train):
        pass

    @abstractmethod
    def evaluate_model(self, model, x_val, y_val):
        pass


class SklearnTemplate(Template):
    def __init__(self, model_cls, metric_func) -> None:
        super().__init__('sklearn')
        self._model_cls = model_cls
        self._metric_func = metric_func

    def build_model(self, hyper_parameters):
        return self._model_cls(**hyper_parameters)

    def fit_model(self, model, x_train, y_train):
        return model.fit(x_train, y_train)

    def evaluate_model(self, model, x_val, y_val):
        y_pred = model.predict(x_val)
        return self._metric_func(y_val, y_pred)


class KerasTemplate(Template):
    def __init__(self,
                 build_func,
                 evaluate_func=None,
                 batch_size=None,
                 epochs=1,
                 callbacks=None,
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None) -> None:
        super().__init__('keras')
        self._build_func = build_func
        self._evaluate_func = evaluate_func
        self._batch_size = batch_size
        self._epochs = epochs
        self._callbacks = callbacks
        self._shuffle = shuffle
        self._class_weight = class_weight
        self._sample_weight = sample_weight

    def build_model(self, hyper_parameters):
        return self._build_func(**hyper_parameters)

    def fit_model(self, model, x_train, y_train):
        return model.fit(
            x_train,
            y_train,
            verbose=0,
            batch_size=self._batch_size,
            epochs=self._epochs,
            callbacks=self._callbacks,
            shuffle=self._shuffle,
            class_weight=self._class_weight,
            sample_weight=self._sample_weight)

    def evaluate_model(self, model, x_val, y_val):
        if self._evaluate_func is None:
            if len(model.metrics_names) != 1:
                raise RuntimeError(
                    'The compiled keras model does not computes any metrics.')
            return model.evaluate(x_val, y_val, verbose=0)[1]

        y_pred = model.predict(x_val)
        return self._evaluate_func(y_val, y_pred)


class Run(NamedTuple):
    metric: float
    model: Any
    hyper_parameters: Dict[str, Any]
    training_duration: float
    evaluation_duration: float


# FIXME: Implement checkpoint during the optimization
# FIXME: Raise error only if checkpoint dir is not empty
class BaseOptimization(ABC):
    def __init__(self,
                 templates: Sequence[Template],
                 search_spaces: Sequence[Dict[str, Any]],
                 minimize: bool = True,
                 checkpoint_dir: str = 'checkpoints',
                 min_checkpoint_interval: int = 10,
                 checkpoint_on_exit: bool = True) -> None:
        if len(templates) != len(search_spaces):
            raise RuntimeError(
                'templates and search_spaces must have the same len')

        self._template = templates
        self._search_space = search_spaces
        self._minimize = minimize

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        else:
            raise RuntimeError(
                f'{checkpoint_dir} already exists and is not a directory')

        self.min_checkpoint_interval = min_checkpoint_interval * 60  # seconds
        self.checkpoint_on_exit = checkpoint_on_exit

    @property
    def log(self) -> List[Run]:
        if not hasattr(self, '_results'):
            raise RuntimeError("You must call 'find_optimum' first.")
        return self._results

    @property
    def best_model(self) -> Run:
        if not hasattr(self, '_results'):
            raise RuntimeError("You must call 'find_optimum' first.")

        best = self._results[0]
        if self._minimize is True:
            for candidate in self._results[1:]:
                if candidate.metric < best.metric:
                    best = candidate
            return best

        for candidate in self._results[1:]:
            if candidate.metric > best.metric:
                best = candidate
        return best

    @property
    def total_duration(self) -> float:
        if not hasattr(self, '_total_duration'):
            raise RuntimeError("You must call 'find_optimum' first.")
        return self._total_duration

    @abstractmethod
    def samples(self,
                search_space: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        pass

    def find_optimum(self, x_train, y_train, x_val, y_val) -> None:
        self._results: List[Run] = []
        templates: List[Template] = []
        begin = time.perf_counter()

        for model_family, search_space in zip(self._template,
                                              self._search_space):
            for hyper_params in self.samples(search_space):
                model = model_family.build_model(hyper_params)

                training_begin = time.perf_counter()
                model_family.fit_model(model, x_train, y_train)
                interruption = time.perf_counter()
                metric = model_family.evaluate_model(model, x_val, y_val)
                evaluation_end = time.perf_counter()

                training_duration = interruption - training_begin
                evaluation_duration = evaluation_end - interruption

                self._results.append(
                    Run(metric, model, hyper_params, training_duration,
                        evaluation_duration))
                templates.append(model_family)

        end = time.perf_counter()
        self._total_duration = end - begin

        if self.checkpoint_on_exit:
            self._checkpoint(templates)

    def _checkpoint(self, templates: List[Template]):
        for idx, run in enumerate(self._results):
            prefix = os.path.join(self.checkpoint_dir, f'_{idx}')
            out_file = prefix + '.out'
            model_file = prefix + '.model'

            template = templates[idx]
            get_loader(template.loader).save_model(run.model, model_file)

            meta = run._asdict()
            meta['model'] = model_file
            meta['loader'] = template.loader

            with open(out_file, 'w') as file:
                json.dump(meta, file, indent=4)


class GridSearch(BaseOptimization):
    def __init__(self,
                 template: Sequence[Template],
                 search_space: Sequence[Dict[str, Any]],
                 minimize: bool = True,
                 checkpoint_dir: str = 'checkpoints',
                 min_checkpoint_interval: int = 10,
                 checkpoint_on_exit: bool = True) -> None:
        super().__init__(template, search_space, minimize, checkpoint_dir,
                         min_checkpoint_interval, checkpoint_on_exit)

    def samples(self,
                search_space: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        hyper_params = search_space.keys()
        for values in itertools.product(*search_space.values()):
            yield dict(zip(hyper_params, values))


class RandomSearch(BaseOptimization):
    def __init__(self,
                 n_iter: int,
                 template: Sequence[Template],
                 search_space: Sequence[Dict[str, Any]],
                 minimize: bool = True,
                 checkpoint_dir: str = 'checkpoints',
                 min_checkpoint_interval: int = 10,
                 checkpoint_on_exit: bool = True) -> None:
        search_space = [to_generator(s) for s in search_space]
        super().__init__(template, search_space, minimize, checkpoint_dir,
                         min_checkpoint_interval, checkpoint_on_exit)
        self.n_iter = n_iter

    def samples(self,
                search_space: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        hyper_params = search_space.keys()
        generators = search_space.values()

        for _ in range(self.n_iter):
            yield dict(zip(hyper_params, [gen() for gen in generators]))


def sampler_from_seq(elements: Sequence[Any]) -> Callable[[], Any]:
    def sampler() -> Any:
        return random.choice(elements)

    return sampler


def to_generator(params: Dict[str, Union[Callable[[], Any], Sequence[Any]]]
                 ) -> Dict[str, Callable[[], Any]]:
    new_params = {}
    for key, value in params.items():
        if isinstance(value, abc.Iterable):
            new_params[key] = sampler_from_seq(value)
        else:
            new_params[key] = value
    return new_params


def load_experiment_results(checkpoint_dir: str) -> List[Run]:
    results: List[Run] = []

    for metafile in glob.glob(f'{checkpoint_dir}/*.out'):
        with open(metafile, 'r') as file:
            meta = json.load(file)
            model_loader = get_loader(meta['loader'])
            model = model_loader.load_model(meta['model'])

            run = Run(meta['metric'], model, meta['hyper_parameters'],
                      meta['training_duration'], meta['evaluation_duration'])
            results.append(run)

    return results
