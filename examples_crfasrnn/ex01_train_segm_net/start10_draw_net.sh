#!/bin/bash

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
lstt="trn val"

for ii in `echo $lstt`
do
    fproto="unet_${ii}.prototxt"
    echo ":: processing : [${fproto}]"
    fimg="${fproto}.png"
    python draw_net.py ${fproto} ${fimg}
done

