load("@ymir_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:private"])

py_library(
    name = 'ymirlib',
    srcs = glob(["ymirlib/**/*.py"]),
    deps = [],
    visibility = ["//tests:__subpackages__"],
)

py_library(
    name = 'datalib',
    srcs = glob(["datalib/**/*.py"]),
    deps = [
        requirement('scikit-learn'),
        requirement('absl-py'),
        requirement('pandas'),
    ],
)

py_library(
    name = 'ymir',
    srcs = glob(["ymir/**/*.py"]),
    deps = [
        'ymirlib',
        'datalib',
        requirement('dm-haiku'),
        requirement('optax'),
        requirement('absl-py'),
        requirement('scikit-learn'),
        requirement('numpy'),
        requirement('hdbscan'),
    ],
    visibility = ["//:__subpackages__"],
)