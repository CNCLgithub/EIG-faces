import numpy as np
import scipy

from utils import load_data
from utils import plotting_tools


class LeastSquaresThreeBases():

    def __init__(self):
        self.N = 175
        self.M = 7
        self.ut = np.triu_indices(self.N,1)
        self.K = len(self.ut[0])
        

        self.bases = np.zeros((4,self.K))
        self.raw_bases = {}
        for i in range(3):
            self.raw_bases[i] = np.zeros((self.N, self.N))

        ### Idealized view-specificity matrix
        canonical_matrix = np.zeros((self.M,self.M))
        canonical_matrix[0,0:self.M] = [1, 0.5, 0, 0, 0, 0, 0]
        canonical_matrix[1,1:self.M] =    [1, 0.5, 0, 0, 0, 0]
        canonical_matrix[2,2:self.M] =   [1, 0.5, 0, 0.5, 0.5]
        canonical_matrix[3,3:self.M] =          [1, 0.5, 0, 0]
        canonical_matrix[4,4:self.M] =               [1., 0, 0]
        canonical_matrix[5,5:self.M] =                [1., 0]
        canonical_matrix[6,6:self.M] =                     [1.]
        matrix = canonical_matrix

        matrix[1:self.M,0] = matrix[0,1:self.M]; matrix[2:self.M,1] = matrix[1,2:self.M]; matrix[3:self.M,2] = matrix[2,3:self.M];
        matrix[4:self.M,3] = matrix[3,4:self.M]; matrix[5:self.M,4] = matrix[4,5:self.M]; matrix[5,6:self.M] = matrix[6:self.M,5];
        for i in range(self.M): 
            matrix[i,i] = 1;
        canonical_matrix_full = np.zeros((self.N,self.N))
        for i in range(7):
            for j in range(7):
                if matrix[i,j] > 0:
                    canonical_matrix_full[i*25:i*25+25,j*25:j*25+25] = matrix[i,j]

        self.bases[0,:] = canonical_matrix_full[self.ut]
        self.raw_bases[0] = canonical_matrix_full

        ## Idealized mirror symmetry matrix
        canonical_matrix[0,0:self.M] = [1, 0.5, 0, 0.5, 1, 0, 0]
        canonical_matrix[1,1:self.M] =  [1, 0.5, 1, 0.5, 0, 0]
        canonical_matrix[2,2:self.M] =   [1, 0.5, 0, 0.5, 0.5]
        canonical_matrix[3,3:self.M] =          [1, 0.5, 0, 0]
        canonical_matrix[4,4:self.M] =               [1, 0, 0]
        canonical_matrix[5,5:self.M] =                  [1, 1]
        canonical_matrix[6,6:self.M] =                     [1]
        matrix = canonical_matrix
        matrix[1:self.M,0] = matrix[0,1:self.M]; matrix[2:self.M,1] = matrix[1,2:self.M]; matrix[3:self.M,2] = matrix[2,3:self.M];
        matrix[4:self.M,3] = matrix[3,4:self.M]; matrix[5:self.M,4] = matrix[4,5:self.M]; matrix[6:self.M,5] = matrix[5,6:self.M];
        for i in range(self.M): 
            matrix[i,i] = 1;
        canonical_matrix_full = np.zeros((self.N,self.N))
        for i in range(3,5):
            for j in range(2):
                if matrix[i,j] > 0:
                    canonical_matrix_full[i*25:i*25+25,j*25:j*25+25] = matrix[i,j]
        for j in range(3,5):
            for i in range(2):
                if matrix[i,j] > 0:
                    canonical_matrix_full[i*25:i*25+25,j*25:j*25+25] = matrix[i,j]

        self.bases[1,:] = canonical_matrix_full[self.ut]
        self.raw_bases[1] = canonical_matrix_full

        ## Idealized View-independence matrix
        paradiagonals = np.zeros((self.N, self.N))
        for i in [24, 49, 74, 99, 124, 149]:
            for j in range(175):
                paradiagonals[j,j] = 1
                paradiagonals[j, np.mod(i+j+1, 175)] = 1
        self.bases[2,:] = paradiagonals[self.ut]
        self.raw_bases[2] = paradiagonals

        ## Background matrix
        tt = np.ones((self.N,self.N))*.5
        self.bases[3,:] = tt[self.ut]
        self.raw_bases[3] = tt

    def do_lstsq_bootstrap_sample(self, matrix, sample):
        N = 175
        matrix = matrix[np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,175))), :]
        matrix = matrix[:, np.hstack((range(50,75), range(25, 50), range(25), range(75,100), range(100,175)))]
        matrix = matrix[sample, :]
        matrix = matrix[:, sample]
        matrix = matrix[self.ut].flatten()

        bases = self.raw_bases.copy()
        for i in range(4):
            bases[i] = self.raw_bases[i][sample, :]
            bases[i] = bases[i][:, sample]
            bases[i] = bases[i][self.ut].flatten()
            bases[i] = bases[i][matrix != 1]

        m = matrix[matrix != 1]
        b = np.zeros((4, len(m)))
        for i in range(4):
            b[i, :] = bases[i]

        return scipy.optimize.nnls(np.transpose(b), m)[0]


    def plot(self, results):
        names = ['latents', 'vgg', 'data']

        normalized = all_vals[:,:,0:3]
        all_highers = all_highers[:,:,0:3]
        all_lowers = all_lowers[:,:,0:3]

        for i in [0, 3, 6]:
            plotting_tools.plot_barplot_views(normalized[int(i/3),:,:], all_highers[int(i/3),:,:], all_lowers[int(i/3),:,:], layers, suffix = names[i/3])
