#!/bin/bash

##export CUDA_VISIBLE_DEVICES=""

penv="${HOME}/python-venv/caffe-crfasrnn/bin/activate"
pcaffe="${HOME}/bin/set-ml-caffe-crfasrnn-bvlc.sh"

################################
if [ -f "$penv" ]; then
    source $penv
fi

if [ -f "$pcaffe" ]; then
    source $pcaffe
fi

################################

python run02_train_model_fcn_v2.py --model_type 'unet' --is_debug

python run02_train_model_fcn_v2.py --model_type 'crfasrnn' --is_debug