# aesir

[![Tests status](https://github.com/codymlewis/aesir/actions/workflows/main.yml/badge.svg)](https://github.com/codymlewis/aesir/actions/workflows/main.yml)
[![License](https://img.shields.io/github/license/codymlewis/aesir?color=blue)](LICENSE)
![Commit activity](https://img.shields.io/github/commit-activity/m/codymlewis/aesir?color=red)

JAX-based Federated learning library focusing on its security.

## Installation

As prerequisite, the `jax` and `jaxlib` libraries must be installed, we omit them from the
included `requirements.txt` as the installed library is respective to the system used. We direct
to first follow https://github.com/google/jax#installation then proceed with this section.

You can either quickly install the package with

```sh
pip install git+https://github.com/codymlewis/aesir.git
```

or build it from source with

```sh
pip install -r requirements.txt
make
```

## Usage

We provide examples of the library's usage in the `samples` folder.