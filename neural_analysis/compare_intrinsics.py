import argparse
import h5py
import pickle
import numpy as np
import scipy
import os

from models.eig.networks.network_classifier import EIG_classifier

from utils.plotting_tools import *
from utils import load_data
from utils import config
CONFIG = config.Config()

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


def main():
    parser = argparse.ArgumentParser(description = "Comparing models and intrinsic image components")

    parser.add_argument('inputfile', type = str, 
                        help = 'Path to the input hdf5 file that contains network output.')
    parser.add_argument('imageset',  type=str,
                        help='BFM (bfm) images or FIV (fiv) images? Important for correctly ordering the matrix columns and rows.')
    
    parser.add_argument('--intrinsicsfolder', type = str, default='./output/intrinsics/',
                        help = 'Path to the folder where intrinsic images are stored. This is the folder where the output of render_intrinsics.m goes.')

    global args
    args = parser.parse_args()
    assert args.imageset in ['bfm', 'fiv'], 'set imageset to either bfm or fiv; e.g., --imageset fiv'

    args.intrinsicsfolder = os.path.join(args.intrinsicsfolder, args.imageset)

    pdf_state = False # Turn this on if you would like the plots to be in pdf (vector graphics) as opposed to png format.

    """
    ### Load neural data and RSA matrices. Compute the RSA matrices if they are not found.
    """
    patches = ['MLMF', 'AL', 'AM']
    neural_vecs = load_data.load_neural_data()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    neural_rsa_file_path = './output/data_similarity.p'
    try:
        data_similarity = pickle.load(open(neural_rsa_file_path, 'rb'))
    except:
        data_similarity = load_data.compute_similarity(neural_vecs, patches)
        pickle.dump(data_similarity, open(neural_rsa_file_path, 'wb'))


    # 
    components = ['Raw', 'Att', 'A', 'N', 'f3']
    #components = ['Raw', 'Att', 'f3']
    imagefolder = os.path.join(CONFIG['PATHS', 'neural'], 'stimuli', args.imageset)
    
    modeling_components = {}
    modeling_components['Raw'] = load_data.load_intrinsics(imagefolder, 'raw')
    modeling_components['A'] = load_data.load_intrinsics(args.intrinsicsfolder, 'albedo')
    modeling_components['N'] = load_data.load_intrinsics(args.intrinsicsfolder, 'normals')
    
    """
    ### Load EIG predictions and RSA matrices
    """
    network_features, _ = load_data.load_model_data_integrated(args.inputfile, args.imageset=='bfm')
    modeling_components['Att'] = network_features['Att']
    modeling_components['f3'] = network_features[0]

    modeling_components_rsa_matrices = load_data.compute_similarity(modeling_components, components)


    """
    ## Bootstrap and compare model and neural RSA matrices
    """
    print('Bootstrapping ...')
    N = 175
    S = 10#000
    patches = ['MLMF']
    comparisons_full = np.zeros((len(components), len(patches), S))
    for i, patch in enumerate(patches):
        for j, comp in enumerate(components):
            print(comp)
            for s in range(S):
                sample = np.random.choice(N, N)
                comparisons_full[j,i,s] = sample_and_similarity(modeling_components_rsa_matrices[comp], data_similarity[patch], sample)

    PT = PlottingTools()
    PT.multiple_barplots(comparisons_full, components, 1, 1, 'modeling-components-to-mlmf', 3.5, 3.5, verbose=True, pdf=pdf_state)


    '''
    ### Show RSA matrices
    '''
    matrices = []
    for i, val in enumerate(components):
        matrix = modeling_components_rsa_matrices[val]
        matrix = matrix[np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N))), :]
        matrix = matrix[:, np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,N)))]
        matrices.append(matrix)

    PT.multiple_rsa_matrices(matrices, 1, len(components), 'model-rsa-matrices', 2, 10, pdf=pdf_state)



if __name__ == '__main__':
    main()
