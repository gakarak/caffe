#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np

# import keras
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

import matplotlib.pyplot as plt

######################################
def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        weight_filler=dict(type='xavier'))
        # ,param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def upsample2d(bottom, nout, psize=2):
    p_kernel_size = 2 * psize - psize % 2
    p_pad = int(np.ceil((psize - 1) / 2.))
    return L.Deconvolution(
        bottom,
        convolution_param = {
            'num_output': nout,
            'kernel_size': p_kernel_size,
            'stride': psize,
            'pad':  p_pad,
            'bias_term': False,
            'weight_filler': {'type': 'bilinear'}
        },
        param = {'lr_mult': 0, 'decay_mult': 0}
    )

######################################
def build_unet(p_pathDataIdx, p_numCls = 2, p_imageSize = 256,
               p_batchSize = 16, p_numFlt = 16,
               p_isInMemory = True, p_isCRFasRNN = False, p_isDeploy = False, p_numChannels = 3):
    net = caffe.NetSpec()
    pydata_params = {
        'img_size':     p_imageSize,
        'batch_size':   p_batchSize,
        'path_idx':     p_pathDataIdx,
        'is_in_memory': p_isInMemory
    }
    if p_isDeploy:
        net.data = L.Input(input_param={
            'shape':
                {
                    'dim':[ p_batchSize, p_numChannels, p_imageSize, p_imageSize]
                }
        })
    else:
        net.data, net.labels = L.Python(module='data_layer_segm',
                                        layer='DataLayerSegmV1', ntop=2, param_str=str(pydata_params))
    # Encoder
    nflt1 = p_numFlt * (2 ** 0)
    net.conv1_1, net.relu1_1 = conv_relu(net.data,    nflt1 , pad=1, ks=3)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, nflt1, pad=1, ks=3)
    net.pool1 = max_pool(net.relu1_2)
    #
    nflt2 = p_numFlt * (2 ** 1)
    net.conv2_1, net.relu2_1 = conv_relu(net.pool1,   nflt2, pad=1, ks=3)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, nflt2, pad=1, ks=3)
    net.pool2 = max_pool(net.relu2_2)
    #
    nflt3 = p_numFlt * (2 ** 2)
    net.conv3_1, net.relu3_1 = conv_relu(net.pool2,   nflt3, pad=1, ks=3)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, nflt3, pad=1, ks=3)
    net.pool3 = max_pool(net.relu3_2)
    #
    nflt4 = p_numFlt * (2 ** 4)
    net.conv4_1, net.relu4_1 = conv_relu(net.pool3,   nflt4, pad=1, ks=3)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, nflt4, pad=1, ks=3)
    net.pool4 = max_pool(net.relu4_2)
    #
    nflt5 = p_numFlt * (2 ** 5)
    net.conv5_1, net.relu5_1 = conv_relu(net.pool4,   nflt5, pad=1, ks=3)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, nflt5, pad=1, ks=3)
    net.pool5 = max_pool(net.relu5_2)
    #
    # Decoder
    nfltup_0 = p_numFlt * (2 ** 2)
    net.upconv0, net.uprelu0 = conv_relu(net.pool5, nfltup_0, pad=0, ks=1)
    #
    nfltup_1 = p_numFlt * (2 ** 1)
    net.upsamle1_x4 = upsample2d(net.uprelu0, nfltup_0, psize=4)
    net.concat1 = L.Concat(net.upsamle1_x4, net.relu4_2)
    net.upconv1, net.uprelu1 = conv_relu(net.concat1, nfltup_1, pad=1, ks=3)
    #
    nfltup_2 = p_numFlt * (2 ** 0)
    net.upsamle2_x4 = upsample2d(net.uprelu1, nfltup_1, psize=4)
    net.concat2 = L.Concat(net.upsamle2_x4, net.relu2_2)
    net.upconv2, net.uprelu2 = conv_relu(net.concat2, nfltup_2, pad=1, ks=3)
    #
    net.upsamle3_x2 = upsample2d(net.uprelu2, nfltup_2, psize=2)
    # net.upconv3, net.score = conv_relu(net.upsamle3_x2, nfltup_3, pad=0, ks=1)
    #
    # Output
    if not p_isCRFasRNN:
        net.score = L.Convolution(net.upsamle3_x2, kernel_size=1, stride=1,
                             num_output=p_numCls, pad=0,
                             weight_filler=dict(type='xavier'))
    else:
        net.coarse_map = L.Convolution(net.upsamle3_x2, kernel_size=1, stride=1,
                             num_output=p_numCls, pad=0,
                             weight_filler=dict(type='xavier'))
        net.split_unary, net.split_q0 = L.Split(net.coarse_map, ntop = 2)
        net.crfmap = L.MultiStageMeanfield(net.split_unary, net.split_q0, net.data,
                                          param = [{'lr_mult': 200}, {'lr_mult': 200}, {'lr_mult': 200}],
                                          multi_stage_meanfield_param = {
                                              'num_iterations': 5,
                                              'compatibility_mode': 0, #POTTS
                                          #     # Initialize the compatilibity transform matrix with a matrix whose diagonal is -1.
                                              'threshold': 2,
                                              'theta_alpha': 160,
                                              'theta_beta': 3,
                                              'theta_gamma': 3,
                                              'spatial_filter_weight': 3,
                                              'bilateral_filter_weight': 5
                                          })
    if p_isDeploy:
        if p_isCRFasRNN:
            net.prob = L.Softmax(net.crfmap)
        else:
            net.prob = L.Softmax(net.score)
    else:
        if p_isCRFasRNN:
            net.loss = L.SoftmaxWithLoss(net.crfmap, net.labels)
            #
            net.prob = L.Softmax(net.crfmap, include={'phase': caffe.TEST})
            net.acc = L.Accuracy(net.prob, net.labels, include={'phase': caffe.TEST})
        else:
            net.loss = L.SoftmaxWithLoss(net.score, net.labels)
            #
            net.prob = L.Softmax(net.score, include={'phase': caffe.TEST})
            net.acc = L.Accuracy(net.prob, net.labels, include={'phase': caffe.TEST})
    return net.to_proto()

######################################
if __name__ == '__main__':
    p_batch_size = 8
    p_image_size = 256
    p_num_cls = 2
    p_is_in_memory = False
    p_is_crfasrnn = True

    modelSpecTrn = build_unet('data/data_trn.csv',
                              p_numCls = p_num_cls,
                              p_imageSize = p_image_size,
                              p_batchSize = p_batch_size,
                              p_isInMemory = p_is_in_memory,
                              p_isCRFasRNN = p_is_crfasrnn)
    with open('unet_trn.prototxt', 'w') as f:
        f.write('{}\n'.format(modelSpecTrn))

    modelSpecVal = modelSpecTrn = build_unet('data/data_val.csv',
                              p_numCls = p_num_cls,
                              p_imageSize = p_image_size,
                              p_batchSize = p_batch_size,
                              p_isInMemory = p_is_in_memory,
                              p_isCRFasRNN=p_is_crfasrnn)
    with open('unet_val.prototxt', 'w') as f:
        f.write('{}\n'.format(modelSpecTrn))
    print (modelSpecTrn)
    print ('-')
    print (modelSpecVal)