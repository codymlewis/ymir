py_binary(
    name = "main",
    deps = ['//:ymir'],
    srcs = ["main.py", "metrics.py", "datasets.py"],
    main = "main.py",
)


py_binary(
    name = "grid_search",
    deps = ['//:ymir'],
    srcs = ["grid_search.py", "metrics.py", "datasets.py"],
    main = "grid_search.py",
)


sh_binary(
    name = "experiments",
    srcs = ["experiments.sh"],
)