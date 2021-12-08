load("@ymir_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    deps = ['//:ymir'],
    srcs = ["main.py"],
    main = "main.py",
)


py_binary(
    name = "plot",
    deps = [requirement('numpy')],
    srcs = ["plot.py"],
    main = "plot.py",
)