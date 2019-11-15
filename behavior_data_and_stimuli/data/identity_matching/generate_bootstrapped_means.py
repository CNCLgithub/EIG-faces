import numpy as np
import pickle
import pandas as pd
import random

def get_split_half_samples(df, wlist, N, S):
    averaged_output = np.zeros((2*N, S))
    indices = list(range(len(wlist)))
    K = int(len(wlist)/2)
    for i in range(N):
        random.shuffle(indices)
        sublist = wlist[indices[0:K]]
        df_mean = pd.pivot_table(df[df.workerid.isin(sublist)], values = ['full_model'], index=['stimulus'], aggfunc=np.mean)
        averaged_output[i*2,:] = df_mean.iloc[:,0]
        sublist = wlist[indices[K:]]
        df_mean = pd.pivot_table(df[df.workerid.isin(sublist)], values = ['full_model'], index=['stimulus'], aggfunc=np.mean)
        averaged_output[i*2+1,:] = df_mean.iloc[:,0]
    return averaged_output

def get_bootstrap_samples(df, wlist, N, S):
    result = np.zeros((N, S))
    K = len(wlist)
    indices = range(K)
    df = df[df.workerid.isin(wlist)]
    for i in range(N):
        new_set = np.random.choice(indices, K)
        df_sample = df[df.workerid == wlist[new_set[0]]]
        frames = []
        for k in range(K):
            frames.append(df[df.workerid == wlist[new_set[k]]])
        df_sample = pd.concat(frames)
        df_mean = pd.pivot_table(df_sample, values = ['full_model'], index=['stimulus'], aggfunc=np.mean)
        result[i,:] = df_mean.iloc[:,0]
    return result

N = 10000
split_half = {}
split_half['exp1'] = np.zeros((2*N, 192))
split_half['exp2'] = np.zeros((2*N, 192))
split_half['exp3'] = np.zeros((2*N, 192))

bootstrap_samples = {}
bootstrap_samples['exp1'] = np.zeros((N, 192))
bootstrap_samples['exp2'] = np.zeros((N, 192))
bootstrap_samples['exp3'] = np.zeros((N, 192))

######## EXP 1
data_dir = './experiment_1/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']

T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index

print('Experiment 1 - num of subjects: '+str(len(wlist)))

split_half['exp1'][:,0:96] = get_split_half_samples(df[df.condition == short], wlist, N, 96)
bootstrap_samples['exp1'][:,0:96] = get_bootstrap_samples(df[df.condition == short], wlist, N, 96)


######## EXP 2
data_dir = './experiment_2/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']

T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index
wlist = wlist[wlist != 'A1VNYP58BTF4HX']
wlist = wlist[wlist != 'A1727A5VJOZAMR']

print('Experiment 2 - num of subjects: '+str(len(wlist)))

split_half['exp2'][:,0:96] = get_split_half_samples(df[df.condition == short], wlist, N, 96)
bootstrap_samples['exp2'][:,0:96] = get_bootstrap_samples(df[df.condition == short], wlist, N, 96)


######## EXP 3
data_dir = './experiment_3/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']

T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index
wlist = wlist[wlist != 'A1727A5VJOZAMR']

print('Experiment 3 - num of subjects: '+str(len(wlist)))

split_half['exp3'][:,0:96] = get_split_half_samples(df[df.condition == short], wlist, N, 96)
bootstrap_samples['exp3'][:,0:96] = get_bootstrap_samples(df[df.condition == short], wlist, N, 96)


pickle.dump(split_half, open('split_half_samples.p', 'wb'))
pickle.dump(bootstrap_samples, open('bootstrap_samples.p', 'wb'))

