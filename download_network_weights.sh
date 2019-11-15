#!/bin/bash

## places alexnet weights
wget -O places365_alexnet_float.pth https://yale.box.com/shared/static/6k70k5pgmlbb92w64le7x1gbxa7mzlpa.pth
mv places365_alexnet_float.pth ./models/raw/

## VRN weights
wget -O vrn_unguided.pth https://yale.box.com/shared/static/tx9w3tccrbpbfcwghuud326ftrb6jdk7.pth
mv vrn_unguided.pth ./models/raw/

## EIG weights (up to f2 without the classification stage)
wget -O checkpoint_bfm.pth.tar https://yale.box.com/shared/static/pz69z9odloncbgahf7h3yydljyqpsi4x
mkdir -p ./models/checkpoints/eig/
mv checkpoint_bfm.pth.tar ./models/checkpoints/eig/

## EIG-classifier weights 
# BFM weights
wget -O checkpoint_bfm.pth.tar https://yale.box.com/shared/static/xt8mq8utemzdwykb5jkpd3vqx70ij4ih
mkdir -p ./models/checkpoints/eig_classifier/
mv checkpoint_bfm.pth.tar ./models/checkpoints/eig_classifier/

# FIV weights
wget -O checkpoint_fiv.pth.tar https://yale.box.com/shared/static/un0p4lyfequvc2shc71x0lbx7f412tup
mv checkpoint_fiv.pth.tar ./models/checkpoints/eig_classifier/

## VGG weights
# VGG raw
wget -O vgg_face_caffe.pth https://yale.box.com/shared/static/vowc9rg78meao7al267s63kgz06nf3z3
mv vgg_face_caffe.pth ./models/raw/

# VGG BFM
wget -O checkpoint_bfm.pth.tar https://yale.box.com/shared/static/9w5lo69ovjsbrhmpia6ucjbn4v7gi2s8
mkdir -p ./models/checkpoints/vgg
mv checkpoint_bfm.pth.tar ./models/checkpoints/vgg/

# VGG FIV
wget -O checkpoint_fiv.pth.tar https://yale.box.com/shared/static/ua7g8ulp28164w81pd87olcg2hzlk24d
mv checkpoint_fiv.pth.tar ./models/checkpoints/vgg/

