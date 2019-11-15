import numpy as np
import pickle
import pandas as pd
import random
from scipy.stats import ttest_ind


######## EXP 1
data_dir = './experiment_1/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']
print('Experiment 1 - num of subjects: '+str(len(np.unique(df.workerid))))
T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index

print('Experiment 1 - num of subjects: '+str(len(wlist)))
print(np.mean(df_mean[df_mean.iloc[:,0] > T]))

ref = df_mean[df_mean.iloc[:,0] > T]


######## EXP 2
data_dir = './experiment_2/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']
print('Experiment 2 - num of subjects: '+str(len(np.unique(df.workerid))))
T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index

print('Experiment 2 - num of subjects: '+str(len(wlist)))
print(np.mean(df_mean[df_mean.iloc[:,0] > T]))
print(ttest_ind(ref, df_mean[df_mean.iloc[:,0] > T]))


######## EXP 3
data_dir = './experiment_3/'
df = pd.read_csv(data_dir+'results.csv', header = 0)
df.columns = ['workerid', 'condition', 'stimulus', 'ground_truth', 'response', 'full_model', 'rt']
print('Experiment 3 - num of subjects: '+str(len(np.unique(df.workerid))))
T = 0.5
short = 150
df_mean = pd.pivot_table(df[df.condition == short], values = ['full_model'], index = ['workerid'], aggfunc = np.mean)
wlist = df_mean[df_mean.iloc[:,0] > T].index

print('Experiment 3 - num of subjects: '+str(len(wlist)))
print(np.mean(df_mean[df_mean.iloc[:,0] > T]))
print(ttest_ind(ref, df_mean[df_mean.iloc[:,0] > T]))

