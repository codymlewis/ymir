#!/bin/sh

dir='~/ymir_datasets'

echo "Getting MNIST"
python mnist.py $dir/mnist
echo

echo "Getting CIFAR 10"
python cifar10.py $dir/cifar10
echo

echo "Getting KDD Cup '99"
python kddcup99.py $dir/kddcup99
echo