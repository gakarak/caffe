#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import caffe

import os
import cv2
import pandas as pd
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

import random

######################################
def load_img_msk(p_img, p_msk, p_img_size = 256, p_num_cls = 2):
    timg = skio.imread(p_img)
    if p_num_cls==2:
        tmsk = (skio.imread(p_msk) > 0).astype(np.uint8)
    else:
        tmsk = (skio.imread(p_msk) - 128).astype(np.uint8)
    if tmsk.ndim>2:
        tmsk = tmsk[:,:,:2]
    #
    if not ((timg.shape[0] == p_img_size) and (timg.shape[1] == p_img_size)):
        timg = cv2.resize(timg, (p_img_size, p_img_size), interpolation=cv2.INTER_CUBIC)
        tmsk = cv2.resize(tmsk, (p_img_size, p_img_size), interpolation=cv2.INTER_NEAREST)
    #
    timg = timg.astype(np.float32) / 255.
    tmsk = np.expand_dims(tmsk.astype(np.float32), axis=0)
    if timg.ndim < 3:
        timg = timg.reshape([1] + list(timg.shape))
    else:
        timg = timg.transpose((2, 0, 1))
    return timg, tmsk

######################################
class DataLayerSegmV1(caffe.Layer):
# class DataLayerSegmV1():
    def setup(self, bottom, top):
        self.img_size = 256
        try:
            params = eval(self.param_str)
            print ('params: {}'.format(params))
            #
            self.img_size = params["img_size"]
            self.batch_size = params["batch_size"]
            self.path_idx = params["path_idx"]
            self.is_in_memory = params["is_in_memory"]
            self.num_cls = params["num_cls"]
        except:
            self.num_cls = 2
            self.img_size = 256
            self.batch_size = 16
            self.path_idx = '/mnt/data1T/@Kaggle/07_Carvana_2017_caffe_x256/data_val.csv'
            self.is_in_memory = False
        print ('param::is_in_memory = {}\nparam::batch_size = {}\nparam::path_idx = {}'
               .format(self.is_in_memory, self.batch_size, self.path_idx))
        self.wdir = os.path.dirname(self.path_idx)
        self.dataCSV = pd.read_csv(self.path_idx)
        self.pathImgs = np.array([os.path.join(self.wdir, xx) for xx in self.dataCSV['path_img']])
        self.pathMsks = np.array([os.path.join(self.wdir, xx) for xx in self.dataCSV['path_msk']])
        self.numSamples = len(self.pathImgs)
        if self.is_in_memory:
            self.dataImg = None
            self.dataMsk = None
            for ii, (pimg, pmsk) in enumerate(zip(self.pathImgs, self.pathMsks)):
                timg, tmsk = load_img_msk(pimg, pmsk, p_img_size = self.img_size, p_num_cls=self.num_cls)
                if self.dataImg is None:
                    self.dataImg = np.zeros([self.numSamples] + list(timg.shape), dtype=np.float32)
                    self.dataMsk = np.zeros([self.numSamples] + list(tmsk.shape), dtype=np.float32)
                self.dataImg[ii] = timg
                self.dataMsk[ii] = tmsk
                if (ii%200) == 0:
                    print ('\t[{}/{}] loading... [{}]'.format(ii, self.numSamples, pimg))
        else:
            self.dataImg = None
            self.dataMsk = None
        #
        self.ridx = np.array(range(self.numSamples))
        if top is not None:
            top[0].reshape(self.batch_size, 3, self.img_size, self.img_size)
            top[1].reshape(self.batch_size, 1, self.img_size, self.img_size)
        self.iter_counter = 0

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        rndIdx = np.random.permutation(self.ridx)[:self.batch_size]
        # print ('rndIdx = {}'.format(rndIdx))

        if self.is_in_memory:
            x_batch = self.dataImg[rndIdx]
            y_batch = self.dataMsk[rndIdx]
        else:
            x_batch = None
            y_batch = None
            for ii, iidx in enumerate(rndIdx):
                pimg = self.pathImgs[iidx]
                pmsk = self.pathMsks[iidx]
                timg, tmsk = load_img_msk(pimg, pmsk, p_img_size=self.img_size, p_num_cls=self.num_cls)
                # print ('img-shape = {}, msk-shape = {}'.format(timg.shape, tmsk.shape))
                if x_batch is None:
                    x_batch = np.zeros([self.batch_size] + list(timg.shape), dtype=np.float32)
                    y_batch = np.zeros([self.batch_size] + list(tmsk.shape), dtype=np.float32)
                x_batch[ii] = timg
                y_batch[ii] = tmsk
        if top is not None:
            top[0].data[...] = x_batch
            top[1].data[...] = y_batch
        self.iter_counter += 1
        if (self.iter_counter%1000) == 0:
            print ('----')
            print ('self.x_data.shape/dtype = {}/{}'.format(x_batch.shape, x_batch.dtype))
            print ('self.y_data.shape/dtype = {}/{}'.format(y_batch.shape, y_batch.dtype))
            print ('----')

    def backward(self, top, propagate_down, bottom):
        pass

######################################
if __name__ == '__main__':
    pathTrn = '/mnt/data1T/@Kaggle/07_Carvana_2017_caffe_x256/data_trn.csv'
    pathVal = '/mnt/data1T/@Kaggle/07_Carvana_2017_caffe_x256/data_val.csv'

    pydata_params = {
        'img_size':     256,
        'batch_size':   32,
        'path_idx':     '/mnt/data1T/@Kaggle/07_Carvana_2017_caffe_x256/data_val.csv',
        'is_in_memory': False
    }

    dataLayer = DataLayerSegmV1()
    dataLayer.param_str = str(pydata_params)
    dataLayer.setup(bottom=None, top=None)

    q1 = dataLayer.forward(top=None, bottom=None)

    print ('-')