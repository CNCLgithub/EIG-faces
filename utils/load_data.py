import os
import numpy as np
import h5py
from scipy.io import loadmat
from PIL import Image
from PIL.ImageOps import autocontrast

from sklearn import preprocessing
from sklearn.decomposition import PCA

from utils import config
CONFIG = config.Config()

N = 175
arr = []
for i in range(7):
    arr = np.hstack((arr, range(i, N, 7)))
arr = np.array(arr, dtype='int16')

def compute_similarity(feature_dict, layers):
    print('calculating similatiry matrices')
    similarity = {}
    for layer in layers:
        if isinstance(layer, list):
            layer = str(layer)
        features = feature_dict[layer]
        N = features.shape[0]
        similarity[layer] = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                similarity[layer][i,j] = np.corrcoef(features[i,:], features[j,:])[0,1]
                    
    return similarity

def load_neural_data():
    #get neural vectors
    patches = ['MLMF', 'AL', 'AM']
    print('loading neural data')
    neural_vecs = {}
    MLMF = loadmat(os.path.join(CONFIG['PATHS', 'neural'], 'data/freiwald_tsao2010', 'neural_vecs_MLMF.mat'))
    MLMF = MLMF['neural_vecs']
    neural_vecs['MLMF'] = MLMF[0:175, :]

    AL = loadmat(os.path.join(CONFIG['PATHS', 'neural'], 'data/freiwald_tsao2010', 'neural_vecs_AL.mat'))
    AL = AL['neural_vecs']
    neural_vecs['AL'] = AL[0:175, :]

    AM = loadmat(os.path.join(CONFIG['PATHS', 'neural'], 'data/freiwald_tsao2010', 'neural_vecs_AM.mat'))
    AM = AM['neural_vecs']
    neural_vecs['AM'] = AM[0:175, :]
    return neural_vecs


def __process_features(features, process):
    '''
    various normalizers one can apply to features.
    '''
    if process == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        features = min_max_scaler.fit_transform(features)
    elif process == 'zscore':
        scaler = preprocessing.StandardScaler(with_mean=True).fit(features)
        features = scaler.transform(features)
    elif process == 'divisive':
        scaler = preprocessing.StandardScaler(with_mean=False).fit(features)
        features = scaler.transform(features)
    elif process == 'norm-l2':
        features = preprocessing.normalize(features, norm='l2')
    elif process == 'norm-l1':
        features = preprocessing.normalize(features, norm='l1')
    elif process == 'pca':
        pca = PCA(n_components=30)
        features = pca.fit_transform(features)
    else:
        features = features

    return features


def load_model_data_integrated(filename, bfm):
    print('loading network activations')
    model_vecs = {}
    layers = []
    f = h5py.File(filename, 'r')
    num_layers = f['number_of_layers'][0]
    for layer in range(num_layers):
        layers.append(layer)
        features = f[str(layer)][()]
        if bfm:
            features = features[arr,:]
        if features.shape[1] == 404:
            features = features[:,:-4] 

        model_vecs[layer] = features

    try:
        features = f['Att'][()]
        layers.append('Att')
        if bfm:
            features = features[arr,:]
        model_vecs['Att'] = features
    except:
        pass

    return model_vecs, layers



def load_intrinsics(imagefolder, component, bfm=True, process='none'):
    N = 175
    im_size = 64
    
    features = []
    for i in range(1, N+1):
        if component == 'albedo' or component == 'normals':
            append = '_' + component
        else:
            append =  ''
        fname = os.path.join(imagefolder, str(i) + append + '.png')
        features.append(_load_image(fname, im_size, component))

    features = np.array(features)
    features = __process_features(features, process)
    
    if component == 'raw' and bfm==False:
        pass
    else:
        features = features[arr,:]

    return features


def _load_image(fname, im_size, component):
    with Image.open(fname).resize((im_size, im_size)) as f:
        image = np.asarray(f, dtype=np.float32)
        
    if component == 'raw':
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, 0:3]        
    else:
        image = image[:, :, 0:3]

    return image.flatten()
