from setuptools import find_packages, setup

import arika

packages = find_packages()
with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()
dependencies = ["scipy~=1.1.0", "joblib~=0.13.2", "tensorflow~=1.12.0"]

setup(
    name=arika.__name__,
    author=arika.__author__,
    author_email='edgar.tx@outlook.com',
    version=arika.__version__,
    url='https://github.com/EdgarTeixeira/arika',
    packages=packages,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=dependencies,
    platforms='any',
    license='MIT')
