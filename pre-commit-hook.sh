#!/bin/sh

isort ymir
yapf -i -p -r ymir
pdoc --math -d restructuredtext -o docs ./ymir
git add -A
