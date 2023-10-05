import numpy as np
import scipy.stats
import pandas as pd

from utilities.visualization import moving_average

def mean_confidence_interval(data, confidence=0.85):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def load_reward_data(paths, moving_window, confidence_interval):
    data = {'mean':[],
            'lower':[],
            'upper':[]}
    
    experiments = [[] for i in range(len(paths))]
    
    for idx, path in enumerate(paths):
        experiment = list(pd.read_csv(f"{path}/plots/data.csv", header=None).iloc[:,1])
        experiments[idx] = moving_average(experiment, moving_window=moving_window)
    
    for i in range(len(experiments[0])):
        array = [a[i] for a in experiments]
        m, l, u = mean_confidence_interval(array, confidence=confidence_interval)
        data['mean'].append(m)
        data['lower'].append(l)
        data['upper'].append(u)
         
    return data
        
    