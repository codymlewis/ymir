#!/bin/sh

isort ymir
yapf -i -p -vv -r ymir
rm -r docs
pdoc --math -d restructuredtext -o docs ymir
git add -A
