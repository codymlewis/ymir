load("@ymir_deps//:requirements.bzl", "requirement")

py_binary(
  name = "fedavg",
  deps = ['//:ymir', requirement('tqdm'),],
  srcs = ["fedavg.py"],
  main = "fedavg.py",
)

py_binary(
  name = "fedmax",
  deps = ['//:ymir', requirement('tqdm'),],
  srcs = ["fedmax.py"],
  main = "fedmax.py",
)

py_binary(
  name = "fedprox",
  deps = ['//:ymir', requirement('tqdm'),],
  srcs = ["fedprox.py"],
  main = "fedprox.py",
)

py_binary(
  name = "parameter_poisoning",
  deps = ['//:ymir', requirement('tqdm'),],
  srcs = ["parameter_poisoning.py"],
  main = "parameter_poisoning.py",
)

py_binary(
  name = "backdoor",
  deps = ['//:ymir', requirement('tqdm'),],
  srcs = ["backdoor.py"],
  main = "backdoor.py",
)