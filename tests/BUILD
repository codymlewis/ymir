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
    name = "coordinate",
    srcs = ['test_coordinate.py'],
    deps = [
        '//:ymir',
        requirement('chex'),
        requirement('absl-py'),
        requirement('dm-haiku'),
        requirement('optax'),
    ],
    main = "test_coordinate.py"
)

py_test(
    name = "collaborator",
    srcs = ['test_collaborator.py'],
    deps = [
        '//:ymir',
        requirement('chex'),
        requirement('absl-py'),
        requirement('dm-haiku'),
        requirement('optax'),
    ],
    main = "test_collaborator.py"
)

py_test(
    name = "mp",
    srcs = ['test_mp.py'],
    deps = [
        '//:ymir',
        requirement('chex'),
        requirement('absl-py'),
        requirement('dm-haiku'),
        requirement('optax'),
    ],
    main = "test_mp.py"
)