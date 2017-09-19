#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import json
import glob
import numpy as np
import pandas as pd
import skimage.io as skio
import matplotlib.pyplot as plt

from optparse import OptionParser

from run01_generate_model_fcn_v1 import build_unet

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2

######################################
def draw_protonet(pathNetProto, pphase="ALL", rankdir='LR'):
    pathImg = '{}.png'.format(pathNetProto)
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(pathNetProto).read(), net)
    if pphase == "TRAIN":
        phase = caffe.TRAIN
    elif pphase == "TEST":
        phase = caffe.TEST
    else:
        phase = None
    caffe.draw.draw_net_to_file(net, pathImg, rankdir, phase)
    return pathImg

######################################
def build_model_and_protobuf(p_pathIdx, p_pathProto,
                             p_imageSize = 256, p_numCls = 2, p_batchSize = 16,
                             p_isIneMemory = True, p_isCRFasRNN = False, p_isDeploy = False):
    modelSpecTrn = build_unet(p_pathIdx,
                              p_numCls = p_numCls,
                              p_imageSize = p_imageSize,
                              p_batchSize = p_batchSize,
                              p_isInMemory = p_isIneMemory,
                              p_isCRFasRNN = p_isCRFasRNN,
                              p_isDeploy = p_isDeploy)
    with open(p_pathProto, 'w') as f:
        f.write('{}\n'.format(modelSpecTrn))

######################################
def printNetInfo(pnet, phase=None):
    infoFeatureMaps = [(k, v.data.shape) for k, v in pnet.blobs.items()]
    infoWeights = [(k, v[0].data.shape) for k, v in pnet.params.items()]
    print ('Info::Feature-Maps [{}] :'.format(phase))
    for ii in infoFeatureMaps: print ('\t{}'.format(ii))
    print ('Info::Weights [{}] :'.format(phase))
    for ii in infoWeights: print ('\t{}'.format(ii))

def check_weights_init(pnet):
    halt_layers = []
    num_layers = len(pnet.params.keys())
    for ii, layer in enumerate(pnet.params.keys()):
        numP = len(pnet.params[layer])
        print ('({}/{}) {}'.format(ii, num_layers, layer))
        for index in range(0, numP):
            # if len(psolver.net.params[layer]) < index + 1:
            #     continue
            tsum = np.sum(pnet.params[layer][index].data)
            if tsum == 0:
                print ('\t{} [{}]:\twsum = {}\tzero!'.format(layer, index, tsum))
            else:
                print ('\t{} [{}]:\twsum = {:0.4f}\t'.format(layer, index, tsum))
                halt_layers.append(layer)
    # print (':: Layers with zero-weights:')
    # for ll in halt_layers_str:
    #     print ('\t{}'.format(ll))
    return halt_layers

######################################
def getLatestModelPath(pprefix):
    lstModels = glob.glob('{}*.caffemodel'.format(pprefix))
    if len(lstModels)<1:
        return None
    else:
        lstIdx = np.array([int(os.path.basename(os.path.splitext(xx)[0]).split('_')[-1]) for xx in lstModels])
        idxLatest = np.argmax(lstIdx)
        pathCaffeModel = lstModels[idxLatest]
        return pathCaffeModel

######################################
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--model_type", dest="model_type", help="model type: [unet, crfasrnn]", default="unet")
    parser.add_option("--is_debug", dest="is_debug", help="debug information", action="store_true", default=True)
    parser.add_option("--is_inmem", dest="is_in_memory", help="load dataset into memory before training",
                      action="store_true", default=False)

    (options, args) = parser.parse_args()
    print ('Options:\n{}'.format(json.dumps(options.__dict__, indent=4)))
    #
    pathIdxTrn = '../datasets/Carvana_2017_x256/data_256x256_trn.csv'
    pathIdxVal = '../datasets/Carvana_2017_x256/data_256x256_val.csv'
    dirSnapshots = 'snapshots'
    if not os.path.isdir(dirSnapshots):
        os.makedirs(dirSnapshots)
    #

    isDebug    = options.is_debug
    isInMemory = options.is_in_memory
    isCRFasRNN = options.model_type == 'crfasrnn'
    prefModel  = 'crfasrnn' if isCRFasRNN else 'unet'
    prefSnapshot = '{}/{}'.format(dirSnapshots, prefModel)
    #
    pathProtoTrn = 'model_{}_trn.prototxt'.format(prefModel)
    pathProtoVal = 'model_{}_val.prototxt'.format(prefModel)
    pathProtoDep = 'model_{}_dep.prototxt'.format(prefModel)
    pathSolver   = 'model_{}_solver.prototxt'.format(prefModel)
    #
    numCls    = 2
    if isCRFasRNN:
        batchSize = 1
    else:
        batchSize = 8
    imageSize = 256
    numEpochs = 10
    numEpochsTest = 1
    numEpochsSnapshot = 1
    numSamplesTrn = len(pd.read_csv(pathIdxTrn))
    numSamplesVal = len(pd.read_csv(pathIdxVal))
    numIterPerEpocnTrn = int(np.ceil(float(numSamplesTrn) / batchSize))
    numIterPerEpochVal = int(np.ceil(float(numSamplesVal) / batchSize))
    #
    param_max_iter      = numEpochs * numIterPerEpocnTrn
    param_test_iter     = numIterPerEpochVal
    param_test_interval = numIterPerEpocnTrn
    param_test_interval = numEpochsTest * numIterPerEpocnTrn
    param_snapshot      = numEpochsSnapshot * numIterPerEpocnTrn
    param_display       = int(np.ceil(float(numIterPerEpocnTrn)/10))
    param_snapshot_prefix = prefSnapshot
    #
    strParamsSolver = """
# Data
train_net: "{}"
test_net:  "{}"

# Solver
base_lr: 0.0001
momentum: 0.9
momentum2: 0.999
lr_policy: "fixed"
type: "Adam"

# Iterations:
display: 10
max_iter: {}

test_iter: {}
test_interval: {}

# snapshot intermediate results
snapshot: {}
snapshot_prefix: "{}"

solver_mode: GPU

test_initialization: false
    
""".format(pathProtoTrn, pathProtoVal,
           param_max_iter,
           param_test_iter, param_test_interval,
           param_snapshot, param_snapshot_prefix)
    with open(pathSolver, 'w') as f:
        f.write(strParamsSolver)
    #
    build_model_and_protobuf(pathIdxTrn, pathProtoTrn,
                             p_imageSize=imageSize, p_batchSize=batchSize, p_numCls=numCls,
                             p_isIneMemory=isInMemory, p_isCRFasRNN=isCRFasRNN, p_isDeploy=False)
    build_model_and_protobuf(pathIdxVal, pathProtoVal,
                             p_imageSize=imageSize, p_batchSize=batchSize, p_numCls=numCls,
                             p_isIneMemory=isInMemory,
                             p_isCRFasRNN=isCRFasRNN, p_isDeploy=False)
    build_model_and_protobuf(None, pathProtoDep,
                             p_imageSize=imageSize, p_batchSize=1, p_numCls=numCls,
                             p_isIneMemory=isInMemory,
                             p_isCRFasRNN=isCRFasRNN, p_isDeploy=True)
    #
    if isDebug:
        plt.subplot(3, 1, 1)
        plt.imshow(plt.imread(draw_protonet(pathProtoTrn, pphase='TRAIN')))
        plt.title('Model: Train')
        plt.subplot(3, 1, 2)
        plt.imshow(plt.imread(draw_protonet(pathProtoVal, pphase='TEST')))
        plt.title('Model: Validation')
        plt.subplot(3, 1, 3)
        plt.imshow(plt.imread(draw_protonet(pathProtoDep)))
        plt.title('Model: Deploy')
        plt.show()
    #
    caffe.set_device(1)
    caffe.set_mode_gpu()
    solver = caffe.AdamSolver(pathSolver)
    #
    print ('----------- [SOLVER] ----------\n----->{}<-----\n\n'.format(strParamsSolver))
    #
    printNetInfo(solver.net, 'TRAIN')
    print('--------\n\n')
    printNetInfo(solver.test_nets[0], 'TEST')

    pathModelLatestUnet = getLatestModelPath(pprefix='{}/unet'.format(dirSnapshots))
    pathModelLatestCRFasRNN = getLatestModelPath(pprefix='{}/crfasrnn'.format(dirSnapshots))

    isModelLoaded = False
    if isCRFasRNN:
        if pathModelLatestCRFasRNN is not None:
            print (' [*] loading weights from: [{}]'.format(pathModelLatestCRFasRNN))
            solver.net.copy_from(pathModelLatestCRFasRNN)
            isModelLoaded = True
        else:
            if pathModelLatestUnet is not None:
                print (' [*] loading weights from: [{}]'.format(pathModelLatestUnet))
                solver.net.copy_from(pathModelLatestUnet)
                isModelLoaded = True
    else:
        if pathModelLatestUnet is not None:
            print (' [*] loading weights from: [{}]'.format(pathModelLatestUnet))
            solver.net.copy_from(pathModelLatestUnet)
            isModelLoaded = True
    #
    check_weights_init(solver.net)
    print ('-')
