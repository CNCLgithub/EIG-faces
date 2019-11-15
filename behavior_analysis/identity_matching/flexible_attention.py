import numpy as np
import h5py
import copy

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


def load_model(path):
    vectors = {}
    counter = 0
    exp = h5py.File(path, 'r')
    for i in range(3):
        exp_in = exp[str(i)][()]
        vectors[i] = exp_in
    exp.close()
    return vectors

def get_sims(preds, w_shape, dist_type=1):
    results = np.zeros(96)
    for i in range(96):
        if w_shape == -1:
            results[i] = np.corrcoef(preds[2*i,0:400], preds[2*i+1,0:400])[0,1]
        else:
            results[i] = np.corrcoef(preds[2*i,0:200], preds[2*i+1,0:200])[0,1]*w_shape + np.corrcoef(preds[2*i,200:400], preds[2*i+1,200:400])[0,1]*(1-w_shape)
    return results


def get_perf(preds):
    #threshold = np.mean(preds)
    threshold = np.median(preds)
    results = preds > threshold
    perf = np.sum(results[:48] == True) + np.sum(results[48:] == False)
    perf = perf / len(preds)
    return perf


def do_flexible_attention(S, path, split_data, bootstrap_d, dist_type=1, free_param=False):
    bootstrap_data = copy.deepcopy(bootstrap_d)
    print('Flexible attention model...')
    ### MODEL
    models = {}
    models[0] = load_model(path)

    behavioral_similarity = np.zeros((3, S))
    weight_lists = np.zeros((3, S))

    store_threshold = np.zeros(3)
    for s in range(S):
        data = get_bootstrap_data(s, bootstrap_data) #consider using split_data for crossvalidation
        for exp in range(3):
            corr = -1
            perf = -1
            if s == 0 or free_param == True:
                for search in np.linspace(0, 1, 50):
                    model = get_sims(models[0][exp], search)
                    if free_param == True:
                        val = np.corrcoef(data[exp], model)[0,1]
                        if val > corr:
                            corr = val
                            weight = search
                    else:
                        val = get_perf(model)
                        if val >= perf:
                            perf = val
                            weight = search
                            store_threshold[exp] = weight
                            corr = np.corrcoef(data[exp], model)[0,1]
            elif s > 0 and free_param == False:
                model = get_sims(models[0][exp], store_threshold[exp])
                weight = store_threshold[exp]
                corr = np.corrcoef(data[exp], model)[0,1]
            
            behavioral_similarity[exp, s] = corr
            weight_lists[exp, s] = weight

    weights = np.nanmean(weight_lists, 1)
    for exp in range(3):
        models[0][exp] = get_sims(models[0][exp], weights[exp])

    return models[0], behavioral_similarity, weight_lists
