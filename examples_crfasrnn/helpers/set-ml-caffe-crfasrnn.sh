#!/bin/bash

bd="$HOME/github/gakarak-caffe.git/build/install"

PKG_CONFIG_PATH="${bd}/lib/pkgconfig:$PKG_CONFIG_PATH"
LD_LIBRARY_PATH="${bd}/lib:$LD_LIBRARY_PATH"
PYTHONPATH="${bd}/python:$PYTHONPATH"
PATH="${bd}/bin:$PATH"


export PATH PKG_CONFIG_PATH LD_LIBRARY_PATH PYTHONPATH


LIBRARY_PATH="${bd}/lib"
CPLUS_INCLUDE_PATH="${bd}/include"

export LIBRARY_PATH CPLUS_INCLUDE_PATH
