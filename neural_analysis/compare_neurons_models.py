import os
import argparse

import h5py
import pickle
import random
import numpy as np
import scipy

from utils import load_data
from utils.plotting_tools import *

from neural_analysis.ls_three_bases import LeastSquaresThreeBases

def sample_and_similarity(network, data, sample):
    N = 175
    network = network[0:N,0:N]
    data = data[0:N,0:N]
    ut = np.triu_indices(N,1)
    # shuffle data and network matrices
    network = network[sample, :]
    network = network[:, sample]
    data = data[sample, :]
    data = data[:, sample]
    network = network[ut].flatten()
    data = data[ut].flatten()
    n = network[network != 1]
    d = data[network != 1]
    return np.corrcoef(n, d)[0,1]

def linear_decomposition_three_bases(layers, similarity, S):
    '''
    Linear decomposition to the three idealized similiarty templates
    '''
    print('Decomposigin to the three bases ... Bootstrapping')
    lstb = LeastSquaresThreeBases()

    ### MODELS
    regression = np.zeros((3, len(layers), S))
    for s in range(S):
        sample = np.random.choice(175, 175)
        normed = np.zeros((len(layers), 3))
        for i, layer in enumerate(layers):
            normed[i] = lstb.do_lstsq_bootstrap_sample(similarity[layer], sample)[:3]
        regression[:, :, s] = np.transpose(normed)

    return regression
    
def main():

    parser = argparse.ArgumentParser(
        description = 'Comparing models and neural populations')
    parser.add_argument('inputfile', type = str, 
                        help = 'Path to the input hdf5 file.')
    parser.add_argument('imageset',  type=str,
                        help='BFM (bfm) images or FIV (fiv) images? Important for correctly ordering the matrix columns and rows.')

    global args
    args = parser.parse_args()

    assert args.imageset in ['bfm', 'fiv'], 'set imageset to either bfm or fiv; e.g., --imageset fiv'

    """
    ### Load neural data and RSA matrices. Compute the RSA matrices if they are not found.
    """
    patches = ['MLMF', 'AL', 'AM']
    if not os.path.exists('./output'):
        os.mkdir('./output')
    neural_rsa_file_path = './output/data_similarity.p'
    try:
        data_similarity = pickle.load(open(neural_rsa_file_path, 'rb'))
    except:
        neural_vecs = load_data.load_neural_data()
        data_similarity = load_data.compute_similarity(neural_vecs, patches)
        pickle.dump(data_similarity, open(neural_rsa_file_path, 'wb'))

    """
    ### Load model predictions and compute RSA matrices.
    """
    network_features, layers = load_data.load_model_data_integrated(args.inputfile, args.imageset == 'bfm')
    network_matrices = load_data.compute_similarity(network_features, layers)

    pdf_state = False # Turn this on if you would like the plots to be in pdf (vector graphics) as opposed to png format.

    """
    ## Bootstrap and compare model and neural RSA matrices
    """
    print('Bootstrapping ...')
    N = 175
    S = 10#000
    comparisons_full = np.zeros((len(patches), len(layers), S))
    for i, patch in enumerate(patches):
        for j, layer in enumerate(layers):
            for s in range(S):
                sample = np.random.choice(N, N)
                comparisons_full[i,j,s] = sample_and_similarity(network_matrices[layer], data_similarity[patch], sample)

    PT = PlottingTools()

    labels = ['MLMF', 'AL', 'AM']
    PT.multiple_barplots(comparisons_full, labels, 1, len(layers), 'general-neurons-to-network', 3.5, 5, verbose=True, pdf=pdf_state)

    """
    ## Linear decomposition of the model RSAs using the idealized similarity templates.
    """
    colors = [(253/255., 174/255., 97/255.), (171/255., 217/255., 233/255.), (44/255., 123/255., 182/255.)]
    labels = None

    # MODEL
    coeffs_model = linear_decomposition_three_bases(layers, network_matrices, S) 
    if args.imageset == 'bfm':
        y_max = 0.7
    else:
        y_max = 0.425
    PT.multiple_barplots(coeffs_model, labels, 1, len(layers), 'model-linear-regress', 3.5, 4., colors=colors, verbose=True, ylimits=(0, y_max), pdf=pdf_state)

    """
    ## Linear decomposition of the neural data RSAs using the idealized similarity templates.
    """
    coeffs_data = linear_decomposition_three_bases(patches, data_similarity, S)
    PT.multiple_barplots(coeffs_data, labels, 1, len(patches), 'data-linear-regress', 3.5, 4., colors=colors, verbose=True, ylimits=(0, 0.425), pdf=pdf_state)


    '''
    Model vs. data scatter plot across all three stages of neural patches
    '''
    colors = [(253/255., 174/255., 97/255.), (171/255., 217/255., 233/255.), (44/255., 123/255., 182/255.)]
    mask = np.zeros(9, dtype='int') # determine which dots are whic color
    mask[3] = 1; mask[4] = 1; mask[5] = 1;
    mask[0] = 2; mask[1] = 2; mask[2] = 2;
    PT.scatter_plot(coeffs_model.mean(axis=2).flatten(), coeffs_data.mean(axis=2).flatten(), 'brain_likeness_all_three_stages', 3.5, 3.5, mask=mask, colors=colors, verbose=True, set_limits=False)

    '''
    ### Show RSA matrices of the model
    '''
    matrices = []
    for i, val in enumerate(layers):
        matrix = network_matrices[val]
        matrix = matrix[np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N))), :]
        matrix = matrix[:, np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N)))]
        matrices.append(matrix)

    if len(layers) == 4:
        PT.multiple_rsa_matrices(matrices, 1, len(layers), 'model-rsa-matrices', 2, 8, pdf=pdf_state)
    else:
        PT.multiple_rsa_matrices(matrices, 1, len(layers), 'model-rsa-matrices', 2, 6, pdf=pdf_state)

    '''
    ### Show RSA matrices of the neural data
    '''
    matrices = []
    for i, val in enumerate(patches):
        matrix = data_similarity[val]
        matrix = matrix[np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N))), :]
        matrix = matrix[:, np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N)))]
        matrices.append(matrix)

    PT.multiple_rsa_matrices(matrices, 1, len(patches), 'data-rsa-matrices', 2, 6, pdf=pdf_state)

if __name__ == '__main__':
    main()

