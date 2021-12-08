load("@ymir_deps//:requirements.bzl", "requirement")

py_test(
    name = "ymirlib",
    srcs = ['test_ymirlib.py'],
    deps = [
        '//:ymirlib',
        requirement('chex'),
        requirement('absl-py'),
    ],
    main = "test_ymirlib.py"
)

py_test(
    name = "scout",
    srcs = ['test_scout.py'],
    deps = [
        '//:ymir',
        requirement('chex'),
        requirement('absl-py'),
        requirement('dm-haiku'),
        requirement('optax'),
    ],
    main = "test_scout.py"
)

py_test(
    name = "distributions",
    srcs = ['test_distributions.py'],
    deps = [
        '//:ymir',
        requirement('chex'),
        requirement('absl-py'),
        requirement('numpy'),
    ],
    main = "test_distributions.py"
)

py_test(
    name = "losses",
    srcs = ['test_losses.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
    ],
    main = "test_losses.py"
)

py_test(
    name = "datasets",
    srcs = ['test_datasets.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
        requirement('chex'),
        requirement('numpy'),
    ],
    main = "test_datasets.py"
)

py_test(
    name = "metrics",
    srcs = ['test_metrics.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
        requirement('chex'),
        requirement('numpy'),
    ],
    main = "test_metrics.py"
)

py_test(
    name = "network",
    srcs = ['test_network.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
    ],
    main = "test_network.py"
)

py_test(
    name = "models",
    srcs = ['test_models.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
        requirement('chex'),
        requirement('dm-haiku'),
    ],
    main = "test_models.py"
)


py_test(
    name = "aggregators",
    srcs = ['test_aggregators.py'],
    deps = [
        '//:ymir',
        requirement('absl-py'),
        requirement('chex'),
    ],
    main = "test_aggregators.py"
)