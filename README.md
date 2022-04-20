# Ymir
[![Tests status](https://github.com/codymlewis/ymir/actions/workflows/main.yml/badge.svg)](https://github.com/codymlewis/ymir/actions/workflows/main.yml)
[![License](https://img.shields.io/github/license/codymlewis/ymir?color=blue)](LICENSE)
![Commit activity](https://img.shields.io/github/commit-activity/m/codymlewis/ymir?color=red)

Tensorflow-based Federated learning library focusing on its security.

## Installation

You can either quickly install the package with

```sh
pip install git+https://github.com/codymlewis/ymir.git
```

or build it from source with

```sh
pip install -r requirements.txt
make
```

## Usage

We provide examples of the library's usage in the `samples` folder. Though, generally
a program involves initializing shared values and the network architecture, then initialization
of our `Captain` object, and finally calling step from that object.
