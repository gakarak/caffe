#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import numpy as np
import pandas as pd
import skimage.io as skio
import matplotlib.pyplot as plt

from data_layer_segm import load_img_msk
from run01_generate_model_fcn_v1 import build_unet
from run02_train_model_fcn_v1 import draw_protonet, build_model_and_protobuf,\
    printNetInfo, check_weights_init, getLatestModelPath

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2

######################################
if __name__ == '__main__':
    pathIdxTrn = '../datasets/Carvana_2017_x256/data_256x256_trn.csv'
    pathIdxVal = '../datasets/Carvana_2017_x256/data_256x256_val.csv'
    dirSnapshots = 'snapshots'
    if not os.path.isdir(dirSnapshots):
        os.makedirs(dirSnapshots)
    #
    isSaveFig  = True
    isDebug    = False
    isInMemory = False
    isCRFasRNN = False
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
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #
    pathModelLatest = getLatestModelPath(pprefix=prefSnapshot)
    #
    net = caffe.Net(pathProtoDep, pathModelLatest, caffe.TEST)
    #
    printNetInfo(net, 'Inference')
    #
    check_weights_init(net)
    print (' [*] weights where loaded from [{}]'.format(pathModelLatest))
    #
    wdir = os.path.dirname(pathIdxVal)
    dataCSVVal = pd.read_csv(pathIdxVal)
    pathImgs = [os.path.join(wdir, xx) for xx in dataCSVVal['path_img']]
    pathMsks = [os.path.join(wdir, xx) for xx in dataCSVVal['path_msk']]

    for ii, (pimg, pmsk) in enumerate(zip(pathImgs, pathMsks)):
        print('[{}/{}] : [{}]'.format(ii, numSamplesVal, pimg))
        timg, tmsk = load_img_msk(pimg, pmsk, p_img_size=imageSize)
        net.blobs['data'].data[0] = timg

        tret = net.forward()['prob']
        tmskPred = np.argsort(-tret[0], axis=0)[0]

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(np.mean(timg, axis=0))
        plt.title('image')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(tmsk[0])
        plt.title('mask-gt')
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(tret[0, 1])
        plt.title('mask-predicted')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(np.dstack((tmsk[0], tmskPred, tmsk[0])))
        plt.title('mask-diff')
        plt.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        if isSaveFig:
            foutFig = '{}-{}.png'.format(pimg, prefModel)
            plt.savefig(foutFig)
        else:
            plt.show()
        print ('-')

