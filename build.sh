#!/usr/bin/env bash

pushd rbrain-core
python3 setup.py bdist_wheel
popd
pushd rbrain-segmenter
python3 setup.py bdist_wheel
popd

