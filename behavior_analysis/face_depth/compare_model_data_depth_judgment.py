import argparse
import numpy as np
import h5py
from scipy.stats import sem

import os

from utils.plotting_tools import *
from utils import config

from scipy.stats import ttest_ind

import pandas as pd

CONFIG = config.Config()

def normalize_each_worker(df):
    """
    z-score per subject data.
    """
    workers = np.unique(df.workerid)
    vector = np.zeros(108)
    for worker in workers:
        a = df[df.workerid == worker]
        a = pd.pivot_table(a, values=['response'], index=['stimulus'])
        a = a - np.mean(a) 
        a = a/np.std(a)
        vector = vector + np.array(a).squeeze()
        
    return vector/len(workers)

def main():
    parser = argparse.ArgumentParser(description='Compare models and data on face depth judgments')
    parser.add_argument('inputfile', type = str, default = '',
                        help='This script requires the txt file containing the predicted depths from a  model. See test.py to obtain such predictions from models.')

    global args
    args = parser.parse_args()

    ### DATA
    L = 9 
    df = pd.read_csv(os.path.join(CONFIG['PATHS', 'behavior'], './data/face_depth/results.csv'), header = 0)
    df.columns = ['workerid', 'condition', 'stimulus', 'response']
    judgments = normalize_each_worker(df)
    N = judgments.shape[0]
    judgments_regular = judgments[np.arange(0, N, 2)]
    judgments_relief = judgments[np.arange(1, N, 2)]

    # MODEL
    with open(args.inputfile) as f:
        perceived = np.loadtxt(f, delimiter=',')

    N = perceived.shape[0]
    perceived = perceived / np.max(perceived)
    perceived = perceived - np.mean(perceived)
    perceived = perceived / np.std(perceived)
    perceived_regular = perceived[np.arange(0, N, 2)]
    perceived_regular /= np.max(perceived)
    perceived_relief = perceived[np.arange(1, N, 2)]
    perceived_relief /= np.max(perceived)

    # Relief (illusory condition)
    N = judgments_regular.shape[0]
    x = np.zeros((3, L))
    sems = np.zeros((3, L))
    for i in range(L):
        vals = judgments_relief[np.arange(i, N, 9)]
        x[0, i] = np.mean(vals)
        sems[0, i] = np.std(vals)

        vals = perceived_relief[np.arange(i, N, 9)]
        x[1, i] = np.mean(vals)
        sems[1, i] = np.std(vals)


    PT = PlottingTools()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    colors = [(228/255.,26/255.,28/255.), 'xkcd:blush',  (55/255.,126/255.,184/255.), 'xkcd:light blue', (0,0,0)]
    PT.multiple_lines(x[:2,:], sems[:2,:], np.array([1, 0.75, 0.50, .25, 0, -0.25, -0.5, -0.75, -1]), 'depth-relief', 2.5, 5, verbose=True, colors=[colors[2], colors[2], colors[1]], linetypes=['--', '-', '-'], markers=['', '', ''], y_limits=(-0.8, 0.8), shade=[True, False])

    print(np.corrcoef(perceived_relief, judgments_relief))
    # Regular (Control condition)
    N = judgments_regular.shape[0]
    x = np.zeros((3, L))
    sems = np.zeros((3, L))
    for i in range(L):
        vals = judgments_regular[np.arange(i, N, 9)]
        x[0, i] = np.mean(vals)
        sems[0, i] = np.std(vals)

        vals = perceived_regular[np.arange(i, N, 9)]
        x[1, i] = np.mean(vals)
        sems[1, i] = np.std(vals)


    PT.multiple_lines(x[:2,:], sems[:2,:], np.array([1.31, 0.98, 0.65, 0.33, 0.00, -0.98, -0.98, -0.98, -0.98]), 'depth-regular', 2.5, 5, verbose=True, colors=[colors[0], colors[0], colors[3]], linetypes=['--', '-', '-'], markers=['', '', ''], y_limits=(-.8, .8), shade=[True, False])

    print(np.corrcoef(perceived_regular, judgments_regular))
    mask = np.zeros(108)
    mask[range(0,108,2)] = 1
    mask = mask.astype(int)
    PT.scatter_plot(judgments, perceived/np.max(perceived), 'trial-level-depth', 3.5, 3.5, mask=mask, colors=[colors[0], colors[2]], verbose=True)

if __name__ == '__main__':
    main()
