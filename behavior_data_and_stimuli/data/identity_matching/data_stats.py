import pandas as pd
import pickle
import h5py

import numpy as np
import random

from tqdm import tqdm

import sys
sys.path.append('/om/user/ilkery/efficient_abs/neural_analysis')
sys.path.append('/om/user/ilkery/efficient_abs/behavioral_pipelines')

from utils import plotting_tools


EXP_COUNT = 3
KEYS = ['exp1', 'exp2', 'exp3']

def load_behavioral_data():
    print('Loading behaivoral data')
    aggregate_data = {}
    data = {}
    T = 0.5
    short = 150

    data_dirs = ['./experiment_1/', './experiment_2/', './experiment_3/']

    for i in range(EXP_COUNT):
        key = KEYS[i]
        data_dir = data_dirs[i]
        df = pd.read_csv(data_dir+'results.csv', header = 0)
        df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']
        df = df[df.condition == short]
        df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
        wlist = df_mean[df_mean.iloc[:,0] > T].index
        if key == 'exp2':
            wlist = wlist[wlist != 'A1VNYP58BTF4HX']
            wlist = wlist[wlist != 'A1727A5VJOZAMR']
        if key == 'exp3':
            wlist = wlist[wlist != 'A1727A5VJOZAMR']
        data[key + '_wlist'] = wlist
        print(key + ' - num of subjects: '+str(len(wlist)))
        # aggregate data
        df_mean = pd.pivot_table(df[df.workerid.isin(wlist)], values = ['full_model'], index=['stimulus'], aggfunc=np.mean)
        aggregate_data[key] = df_mean.iloc[:, 0]
        aggregate_data[key][48:96] = 1 - aggregate_data[key][48:96] # probability of "same" response.
        # individual data
        df_mean = df[df.workerid.isin(wlist)]
        data[key] = df_mean

    return data, aggregate_data

def get_bootstrap_data_by_trial(data, models, model_count):
    train_data = {}
    test_data = {}

    test_models = {}
    train_models = {}
    for i in range(model_count):
        test_models[i] = np.zeros((EXP_COUNT, Test*2, models[i][0].shape[1]))
        train_models[i] = np.zeros((EXP_COUNT, Train*2, models[i][0].shape[1]))

    for j in range(EXP_COUNT):
        indices_same = range(48)
        indices_diff = range(48, 96)
        random.shuffle(indices_same)
        random.shuffle(indices_diff)
        train_indices = np.hstack([indices_same[:Train/2], indices_diff[:Train/2]])
        test_indices = np.hstack([indices_same[Train/2:], indices_diff[Train/2:]])
        #train_indices = np.random.choice(train_indices, Train)
        test_indices = np.random.choice(test_indices, Test)

        train_data[KEYS[j]] = data[KEYS[j]][train_indices+1]
        test_data[KEYS[j]] = data[KEYS[j]][test_indices+1]
        for i in range(model_count):
            for k in range(Train):
                train_models[i][j, 2*k, :] = models[i][j][2 * train_indices[k], :]
                train_models[i][j, 2*k + 1, :] = models[i][j][2 * train_indices[k] + 1, :]
            for k in range(Test):
                test_models[i][j, 2*k, :] = models[i][j][2 * test_indices[k], :]
                test_models[i][j, 2*k + 1, :] = models[i][j][2 * test_indices[k] + 1, :]
            
    return train_data, test_data, train_models, test_models


def get_bootstrap_data(data, models, model_count):
    Test = 96
    test_data = {}

    for j in range(EXP_COUNT):
        exp_data = data[KEYS[j]]
        wlist = data[KEYS[j] + '_wlist']
        indices = range(len(wlist))
        indices = np.random.choice(indices, len(wlist))
        frames = []
        df_sample = exp_data[exp_data.workerid == wlist[indices[0]]]
        for k in range(len(wlist)):
            frames.append(exp_data[exp_data.workerid == wlist[indices[k]]])
        df_sample = pd.concat(frames)
        df_mean = pd.pivot_table(df_sample, values=['full_model'], index=['stimulus'], aggfunc=np.mean)
        test_data[KEYS[j]] = df_mean.iloc[:,0]
        test_data[KEYS[j]][48:] = 1 - test_data[KEYS[j]][48:]
            
    return test_data, test_data, models, models

data, aggregate_data = load_behavioral_data()


### DIFFERENT TRIALS PR(SAME) DISTRIBUTION


for key in KEYS:

    a = data[key]
    a = pd.pivot_table(a, values=['full_model'], index=['workerid'], aggfunc=np.mean)
    print('Accuracy')
    print(key+' stats: mean, std, min, max')
    print([a.mean(), a.std(), a.min(), a.max()])

    a = data[key]
    a = pd.pivot_table(a, values=['rt'], index=['workerid'], aggfunc=np.mean)
    print('RT')
    print(key+' stats: mean, std, min, max')
    print([a.mean(), a.std(), a.min(), a.max()])


### learning curves

lr = np.zeros((96, EXP_COUNT, 44))
for key_index, key in enumerate(KEYS):
    a = data[key + '_wlist']
    for w_index in range(44):
        w = a[w_index]
        print(np.array(data[key][data[key].workerid == w].full_model).shape)
        lr[:, key_index, w_index] = np.array(data[key][data[key].workerid == w].full_model)[0:96]


STRIDE = 1
KERNEL = 19
agg_lr = np.zeros((96 - KERNEL + 1, EXP_COUNT, 44))
for i in range(EXP_COUNT):
    for j in range(96 - KERNEL + 1):
        agg_lr[j, i, :] = np.sum(lr[j:(j+KERNEL), i, :], axis=0)/KERNEL



plotting_tools.plot_learning_curves(agg_lr, x_list=range(96 - KERNEL + 1), num_tr=EXP_COUNT, suffix='learning curves', errorbar=True)




"""


print('Bootstrap starting ...\n')
for s in tqdm(range(SAMPLE_COUNT)):
    train_data, test_data, train_models, test_models = get_bootstrap_data(data, models, MODEL_COUNT)
    j = 0 # Selective attention
    for tr in range(EXP_COUNT):
        w_shape = infer_w_shape(train_data[KEYS[tr]], train_models[j][tr])
        model = get_sims(test_models[j][tr], 1, True, w_shape)
        rr_lists[j, tr, s] = np.corrcoef(test_data[KEYS[tr]], model)[0, 1]
        weight_lists[tr, 0, s] = w_shape

    for j in range(1, MODEL_COUNT):
        for tr in range(EXP_COUNT):
            model = get_sims(test_models[j][tr], 1, out_types[j])
            rr_lists[j, tr, s] = np.corrcoef(test_data[KEYS[tr]], model)[0, 1]

plotting_tools.plot_behavior(rr_lists, x_list=range(len(paths)), num_tr=EXP_COUNT, suffix='three experiments', labels=['EIG', 'VGG', 'VGG-FT', 'ID'])

print('weight coefficients')
plotting_tools.plot_behavior(weight_lists, x_list=range(3), num_tr=1, suffix='weights', labels=['Exp1', 'Exp2', 'Exp3'])
"""
