import os
import argparse

import numpy as np
import h5py
from scipy.stats import sem

from utils.plotting_tools import *
from utils import config

from scipy.stats import ttest_ind
import pandas as pd

CONFIG = config.Config()

def normalize_each_worker(df):
    workers = np.unique(df.workerid)
    vector = np.zeros(45)

    for worker in workers:
        a = df[df.workerid == worker]
        a = pd.pivot_table(a, values=['response'], index=['stimulus'])
        a = a - np.mean(a) 
        a = a/np.std(a)
        vector = vector + np.array(a).squeeze()
    return vector/len(workers)


def main():
    parser = argparse.ArgumentParser(description='Compare models and data on lighting direction judgments')
    parser.add_argument('inputfile', type=str, help='This script requires the txt filef containing the predicted depths from a  model. See test.py to obtain such predictions from models.')

    global args
    args = parser.parse_args()

    means = {}; sems = {}; all_vectors = {}
    L = 9
    truth = np.zeros((2,L))
    # elevation
    truth[0,:] = np.array([1.31, 0.98, 0.65, 0.33, 0, -0.33, -0.65, -0.98, -1.31])
    # relief
    truth[1,:] = np.array([1.31]*L)

    ### DATA
    means = {}
    sems = {}
    #regular
    df = pd.read_csv(os.path.join(CONFIG['PATHS', 'behavior'], 'data/lighting_direction/results.csv'), header = 0)
    df.columns = ['workerid', 'condition', 'stimulus', 'response']
    all_vectors[2] = normalize_each_worker(df[df.condition == 0])
    regular_mean = np.zeros(L)
    regular_sem = np.zeros(L)
    for i in range(L):
        vals = all_vectors[2][(range(i*5,(i+1)*5))]
        regular_mean[i] = np.mean(vals)
        regular_sem[i] = np.std(vals)
    means[2] = regular_mean
    sems[2] = regular_sem

    #relief
    all_vectors[3] = normalize_each_worker(df[df.condition == 1])
    relief_mean = np.zeros(L)
    relief_sem = np.zeros(L)
    for i in range(L):
        vals = all_vectors[3][(range(i*5,(i+1)*5))]
        relief_mean[i] = np.mean(vals)
        relief_sem[i] = np.std(vals)
    means[3] = relief_mean
    sems[3] = relief_sem


    #### MODEL
    f = h5py.File(args.inputfile)
    # regular
    results = np.array(f['0'].value, dtype='float64')
    lights = results[:,-1]
    lights = lights[0:45]
    lights = lights - np.mean(lights)
    lights = lights / np.std(lights)
    lights = lights / np.max(lights)
    all_vectors[0] = lights[0:45]
    lights_mean = np.zeros(L)
    lights_sem = np.zeros(L)
    for i in range(L):
        vals = lights[range(i*5,(i+1)*5)]
        lights_mean[i] = np.mean(vals)
        lights_sem[i] = np.std(vals)
    means[0] = lights_mean
    sems[0] = lights_sem

    # relief
    results = np.array(f['1'].value, dtype='float64')
    f.close()
    lights = results[:,-1]
    lights = lights[0:45]
    lights = lights - np.mean(lights)
    lights = lights / np.std(lights)
    lights = lights / np.max(lights)
    all_vectors[1] = lights[0:45]
    lights_mean = np.zeros(L)
    lights_sem = np.zeros(L)
    for i in range(L):
        vals = lights[range(i*5,(i+1)*5)]
        lights_mean[i] = np.mean(vals)
        lights_sem[i] = np.std(vals)
    means[1] = lights_mean
    sems[1] = lights_sem
    eig_means = means.copy()
    eig_sems = sems.copy()

    PT = PlottingTools()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    colors = [(228/255.,26/255.,28/255.), 'xkcd:blush',  (55/255.,126/255.,184/255.), 'xkcd:light blue', (0,0,0)]
    dashes = ['dash --', 'solid -']

    z_means = []
    z_sems = []
    z_means.append(means[2])      # data
    z_sems.append(sems[2])
    z_means.append(means[0])      # model
    z_sems.append(sems[0])
    z_means.append(truth[0])      # ground truth
    z_sems.append(np.zeros(L))
    z_means = np.array(z_means)
    z_sems = np.array(z_sems)
    PT.multiple_lines(z_means, z_sems, np.array([1.31, 0.98, 0.65, 0.33, 0.00, -0.33, -0.65, -0.98, -1.31]), 'elevation', 2.5, 5, verbose=True, colors=[colors[0], colors[0], colors[4]], linetypes=['--', '-', '-'], markers=['', '', '.'], y_limits=(-1.98, 1.63), shade=[True, False, False])


    z_means = []
    z_sems = []
    z_means.append(means[3])      # data
    z_sems.append(sems[3])
    z_means.append(means[1])      # model
    z_sems.append(sems[1])
    z_means.append(truth[1])      # ground truth
    z_sems.append(np.zeros(L))
    z_means = np.array(z_means)
    z_sems = np.array(z_sems)
    PT.multiple_lines(z_means, z_sems, np.array([1, 0.75, 0.50, .25, 0, -0.25, -0.5, -0.75, -1]), 'depth', 2.5, 5,  verbose=True, colors=[colors[2], colors[2], colors[4]], linetypes=['--', '-', '-'], markers=['', '', '.'], y_limits=(-1.98, 1.63), shade=[True, False, False])

    #regular
    print(np.corrcoef(np.array(all_vectors[2]), np.array(all_vectors[0])))
    #relief
    print(np.corrcoef(np.array(all_vectors[3]), np.array(all_vectors[1])))

    data = np.hstack([np.array(all_vectors[2]), np.array(all_vectors[3])])
    model = np.hstack([np.array(all_vectors[0]), np.array(all_vectors[1])])

    mask = np.zeros(model.shape)
    mask[0:45] = 1
    mask = mask.astype(int)

    PT.scatter_plot(data, model, 'trial-level-lighting', 3.5, 3.5, mask=mask, colors=[colors[0], colors[2]], verbose=True)


if __name__ == '__main__':
    main()
