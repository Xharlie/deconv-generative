#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=deconv_solver.prototxt $@