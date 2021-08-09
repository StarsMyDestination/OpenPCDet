#!/usr/bin/env bash

export MAX_JOBS=1 # avoid too much resource, ref to https://pytorch.org/docs/stable/cpp_extension.html
python setup.py develop | tee ./start.log
