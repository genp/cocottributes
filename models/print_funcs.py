import numpy as np

def print_layer_params(solver, weights):
    '''
    wieghts should be taken from same solver in subsequent calls to this function with same wieght dict
    '''
    for key, value in solver.net.params.items():
        if key not in weights.keys():
            weights[key] = {}
        for k2, v2 in enumerate(value):
            if k2 not in weights[key].keys():
                weights[key][k2] = v2.data.ravel()[:100]
            else:
                weights[key][k2] = np.vstack([weights[key][k2], v2.data.ravel()[:100]])
