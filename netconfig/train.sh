#!/bin/bash
/home/ubuntu/caffe-master/build/tools/caffe train -solver=solver.prototxt -weights=netmodel/bvlc_reference_caffenet.caffemodel -gpu 0 2>&1 | tee log/train.log
