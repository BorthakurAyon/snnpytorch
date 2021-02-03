from setuptools import setup, find_packages
from snnpytorch import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="snnpytorch",
    version=__version__,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Ayon Borthakur",
    author_email="ab2535@cornell.edu",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "scipy",
        "pandas",
        "ruamel.yaml",
        "torch",
        "torchvision",
    ],
    extras_require={
        'dev': [
            'pytest>=3.6', 'pytest-cov', 'flake8', 'sphinx-rtd-theme',
            'coveralls', 'sphinx'],
    },
)
