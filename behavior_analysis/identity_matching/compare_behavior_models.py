import argparse
import os

import pickle
import numpy as np
from scipy.io import loadmat
import h5py
import cv2

from behavior_analysis.identity_matching import flexible_attention
from utils import load_data
from utils.plotting_tools import *

from utils import config
CONFIG = config.Config()

def get_bootstrap_data(s, bootstrap_data):
    data = {}
    exp1 = bootstrap_data['exp1'][s,0:96]
    exp1[48:96] = 1-exp1[48:96]
    exp2 = bootstrap_data['exp2'][s,0:96]
    exp2[48:96] = 1-exp2[48:96]
    exp3 = bootstrap_data['exp3'][s,0:96]
    exp3[48:96] = 1-exp3[48:96]

    data[0] = exp1
    data[1] = exp2
    data[2] = exp3

    return data

def get_data_correlation(s, split_data):

    corrs = np.zeros(3)

    exp1 = split_data['exp1'][2*s,0:96]
    exp1[48:96] = 1-exp1[48:96]
    exp1_ = split_data['exp1'][2*s+1,0:96]
    exp1_[48:96] = 1-exp1_[48:96]
    corrs[0] = np.corrcoef(exp1, exp1_)[0,1]

    exp2 = split_data['exp2'][2*s,0:96]
    exp2[48:96] = 1-exp2[48:96]
    exp2_ = split_data['exp2'][2*s+1,0:96]
    exp2_[48:96] = 1-exp2_[48:96]
    corrs[1] = np.corrcoef(exp2, exp2_)[0,1]

    exp3 = split_data['exp3'][2*s,0:96]
    exp3[48:96] = 1-exp3[48:96]
    exp3_ = split_data['exp3'][2*s+1,0:96]
    exp3_[48:96] = 1-exp3_[48:96]
    corrs[2] = np.corrcoef(exp3, exp3_)[0,1]

    return corrs

def cos_angle(u,v):
    return np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v)

def get_sims(preds, dist_type):
    results = np.zeros(96)
    for i in range(96):
        if dist_type == 1:
            results[i] = np.corrcoef(preds[2*i], preds[2*i+1])[0,1]
        elif dist_type == 2:
            results[i] = -1*np.linalg.norm(preds[2*i] - preds[2*i+1])
        elif dist_type == 3:
            results[i] = -cv2.compareHist(preds[2*i], preds[2*i+1], cv2.HISTCMP_CHISQR)
        else:
            results[i] = cos_angle(preds[2*i], preds[2*i+1])
    return results

def load_model(path, dist_type):
    sims = np.zeros((3, 96))
    counter = 0
    exp = h5py.File(path, 'r')
    for i in range(3):
        exp_in = exp[str(i)][()]
        sims[i,:] = get_sims(exp_in, dist_type)
    exp.close()
    return sims


parser = argparse.ArgumentParser(description='Test behavior (recognition)')
parser.add_argument('--make-figure',  type = int, default=1,
                    help='whether to plot results')
parser.add_argument('--save-eig', type = int, default = 0,
                    help='whether to save output of eig. If >0 means epoch number.')

global args
args = parser.parse_args()

### MODELS
models = {}
model_names = ['eig_classifier', 'vgg_raw', 'vgg', 'sift', 'pixels']

filenames_d = {
    'eig' : 'eig.hdf5',
    'eig_classifier' : 'eig_classifier.hdf5',
    'vgg' : 'vgg.hdf5',
    'vgg_raw' : 'vgg_raw.hdf5',
    'pixels': 'image_matching_pixels.hdf5',
    'sift': 'image_matching_sift.hdf5'
}

dist_type = 1
for i, model in enumerate(model_names):
    path = './output/' + filenames_d[model]
    print('loading model ' + path)
    if model == 'sift':
        models[i] = load_model(path, 3)
    else:
        models[i] = load_model(path, dist_type)
print('all models are loaded')

S = 10000
MODEL_COUNT = len(model_names)
with open(os.path.join(CONFIG['PATHS', 'behavior'], 'data/identity_matching/split_half_samples.p'), 'rb') as f:
    split_data = pickle.load(f, encoding='latin1')
with open(os.path.join(CONFIG['PATHS', 'behavior'], 'data/identity_matching/bootstrap_samples.p'), 'rb') as f:
    bootstrap_data = pickle.load(f, encoding='latin1')

behavioral_similarity = np.zeros((MODEL_COUNT, 3, S))

### do the flexible attention model, EIG*

models[0], behavioral_similarity[0], weight_list = flexible_attention.do_flexible_attention(S, './output/' + filenames_d['eig_classifier'], split_data, bootstrap_data, free_param=True)


for s in range(S):
    data = get_bootstrap_data(s, bootstrap_data)

    for j in range(1, MODEL_COUNT): #exclude EIG here
    #for j in range(0, MODEL_COUNT): #exclude EIG here
        for exp in range(3):
            model = models[j][exp]
            behavioral_similarity[j, exp, s] = np.corrcoef(data[exp], model)[0,1]


for i in range(MODEL_COUNT):
    print('model '+ model_names[i])
    print(np.nanmean(behavioral_similarity[i,:,:],1))

data = np.zeros((3,96))
exp1 = (split_data['exp1'][0,0:96] + split_data['exp1'][1,0:96])/2.
exp1[48:96] = 1-exp1[48:96]
exp2 = (split_data['exp2'][0,0:96] + split_data['exp2'][1,0:96])/2.
exp2[48:96] = 1-exp2[48:96]
exp3 = (split_data['exp3'][0,0:96] + split_data['exp3'][1,0:96])/2.
exp3[48:96] = 1-exp3[48:96]
data[0] = exp1
data[1] = exp2
data[2] = exp3

#plotting_tools.plot_behavior_scatter(models, num_tr=num_tr, suffix='fancy', model_count=3)

colors = [(228/255., 26/255., 28/255.), (221/255., 28/255., 119/255.), (201/255., 148/255., 199/255), (77/255., 77/255., 77/255.), (122/255., 122/255., 122/255.)]
if args.make_figure == 1:
    PT = PlottingTools()
    labels = ['EIG', 'VGG-\nRaw', 'VGG', 'SIFT', 'Pixels']
    PT.multiple_barplots(behavioral_similarity, labels, 1, 3, 'behavioral-similarity', 3., 5.25, colors=colors, verbose=False, ylimits=(-0.1, 0.8))

    labels = ['Exp1', 'Exp2', 'Exp3']
    PT.multiple_barplots(np.expand_dims(weight_list, 1), labels, 1, 1, 'shape-bias', 3., 2, verbose=False, ylimits=(0, 1.1))
    print(np.mean(weight_list, 1))

if args.save_eig > 0:
    mean_corr = np.nanmean(behavioral_similarity[0,:,:],1)
    np.save('./output/eig_monitor_' + str(args.save_eig), mean_corr)

"""
print(np.sum(behavioral_similarity[0,0,:] - behavioral_similarity[2,0,:] < 0))
print(np.sum(behavioral_similarity[0,2,:] - behavioral_similarity[2,2,:] < 0))
print(np.sum(behavioral_similarity[0,2,:] - behavioral_similarity[1,2,:] < 0))
"""
