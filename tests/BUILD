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