#!/usr/bin/env bash

bazel build research/viceroy/main

for alg in foolsgold krum std_dagmm viceroy; do
    for attack in 'onoff labelflip' 'scaling backdoor' 'onoff freerider' 'bad mouther' 'good mouther'; do
        for dataset in mnist kddcup99 cifar10; do
            for aper in 0.1 0.3 0.5 0.8; do
                ./bazel-bin/research/viceroy/main --alg "$alg" --attack "$attack" --dataset "$dataset" --aper "$aper"
            done
        done
    done
done