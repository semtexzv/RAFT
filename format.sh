#!/usr/bin/env bash

for f in `find -name "*.py"`; do autopep8 --in-place --select=E,W $f; done