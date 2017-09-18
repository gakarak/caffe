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
gpuid=1
mdl='unet'
sdir='snapshots'
p_solver="model_${mdl}_solver.prototxt"

mkdir -p "${sdir}"

if [ -d "${sdir}" ]; then
    midx=`find ${sdir} -name "${mdl}*.solverstate" | awk -F\_iter\_ '{ print $2 }' | cut -d\. -f1 | sort -n | tail -n 1`
    echo "midx = [${midx}]"
    if [ -n "${midx}" ]; then
	preset=`ls -1 ${sdir}/${mdl}*${midx}*.solverstate | head -n 1`
    fi
fi

echo "reset file = [${preset}]"

if [ -z "${preset}" ]; then
    echo " !!!! Cant find reset file in dir: [${mdir}]"
    sleep 1
    caffe train -solver $p_solver -gpu $gpuid
else
    echo "**** Resuming model from : [${preset}]"
    sleep 1
    caffe train -solver $p_solver -snapshot $preset -gpu $gpuid
fi

##caffe train -solver unet_solver_sgd.prototxt -gpu 1
##caffe train -solver model_${mdl}_solver.prototxt -gpu 1

