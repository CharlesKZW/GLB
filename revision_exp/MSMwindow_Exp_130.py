#%%
import numpy as np
import pandas as pd
import time as time
import sys

from ExpLB import Bounded1NN


# read datasets

dataset_names_df = pd.read_excel('UCR_data_summary.xlsx',
                                engine='openpyxl', 
                                usecols=(2, 6))
dataset_names = np.array([dataset_names_df])
dataset_names = np.squeeze(dataset_names)

dict = {} 
dataset_list = dataset_names[0:130, 0]

path = 'UCR2018-NEW/'

for d in dataset_list:

    k = "{}_train".format(d)
    df = pd.read_csv(path +'{}/{}_TRAIN'.format(d, d), header = None)
    value = df.values
    dict[k] = value

    k = "{}_train_X".format(d) 
    value = dict['{}_train'.format(d)][:, 1:]
    dict[k] = value

    k = "{}_train_y".format(d) 
    value = dict['{}_train'.format(d)][:, 0]
    dict[k] = value
    
    k = "{}_test".format(d) 
    df = pd.read_csv(path +'{}/{}_TEST'.format(d, d), header = None)
    value = df.values
    dict[k] = value
    
    k = "{}_test_X".format(d) 
    value = dict['{}_test'.format(d)][:, 1:]
    dict[k] = value
    
    
    k = "{}_test_y".format(d) 
    value = dict['{}_test'.format(d)][:, 0]
    dict[k] = value

# Experiments






dataset_test = dataset_names[120:130, 0]


full_algo_list = ["LB_Keogh",  "GLB_DTW", "LB_Kim", "LB_New", "LB_Improved", 
             "LB_Keogh_LCSS", "GLB_LCSS", 
             "LB_Kim_ERP", "LB_Keogh_ERP", "GLB_ERP", "LB_ERP", 
             "LB_MSM", "GLB_MSM",
             "LB_TWED", "GLB_TWED",
             "GLB_EDR",
             "GLB_SWALE",
             "Cas_Keogh_GLB", "Cas_Keogh_New", "Cas_Keogh_Improved",
             "Breakdown_QueryData", "Breakdown_QueryOnly", "Breakdown_QueryBoundary"]

algo_list = ["LB_MSM", "GLB_MSM"]

results = {}

WARPING_WINDOW_LIST = [0.05, 0.1, 0.5, 0.75, 1.00]

for d in dataset_test:

    length_of_dataset = dict["{}_test_X".format(d)].shape[1]
    dataset_result = {}

    for algo in algo_list:

        for WARPING_WINDOW in WARPING_WINDOW_LIST:
    
            ticc = time.time()
            if algo == "GLB_EDR":
                m = Bounded1NN(metric = "{}".format(algo), 
                            lb = True, 
                            constraint = "Sakoe-Chiba", 
                            w = WARPING_WINDOW, 
                            epsilon = 0.1, 
                            m = 0, 
                            g = 0.3, 
                            c =0.5, 
                            lamb =1, 
                            nu = 0.0001, 
                            timesx = None, timesy = None, 
                            p = 5, r = 1)
            else:
                m = Bounded1NN(metric = "{}".format(algo), 
                            lb = True, constraint = "Sakoe-Chiba", 
                            w = WARPING_WINDOW, 
                            epsilon = 0.2, 
                            m = 0, 
                            g = 0.3, 
                            c =0.5, 
                            lamb =1, 
                            nu = 0.0001, 
                            timesx = None, timesy = None, 
                            p = 5, r = 1)
            m.fit(dict["{}_train_X".format(d)], dict["{}_train_y".format(d)])
            lb_label, pruning_power = m.predict(dict["{}_test_X".format(d)])
        
            lb_diff = lb_label - dict["{}_test_y".format(d)]
            lb_wrong_predict_count = np.count_nonzero(lb_diff)
            lb_acc = (dict["{}_test_y".format(d)].shape[0] - lb_wrong_predict_count) / dict["{}_test_y".format(d)].shape[0]
            tocc = time.time()
            dataset_result['Bounded1NN_{}{}_acc'.format(algo, WARPING_WINDOW)] = lb_acc
            dataset_result['Bounded1NN_{}{}_pruning'.format(algo, WARPING_WINDOW)] = pruning_power
            dataset_result['Bounded1NN_{}{}_runtime'.format(algo, WARPING_WINDOW)] = tocc - ticc

        
            ticc = time.time()
            if algo == "GLB_EDR":
                m = Bounded1NN(metric = "{}".format(algo), 
                            lb = False, 
                            constraint = "Sakoe-Chiba", 
                            w = WARPING_WINDOW, 
                            epsilon = 0.1, 
                            m = 0, 
                            g = 0.3, 
                            c =0.5, 
                            lamb =1, nu = 0.0001, 
                            timesx = None, timesy = None, 
                            p = 5, r = 1)
            else:
                m = Bounded1NN(metric = "{}".format(algo), 
                            lb = False, 
                            constraint = "Sakoe-Chiba", 
                            w = WARPING_WINDOW, 
                            epsilon = 0.2, 
                            m = 0, 
                            g = 0.3, 
                            c =0.5, 
                            lamb =1, 
                            nu = 0.0001, 
                            timesx = None, timesy = None, 
                            p = 5, r = 1)

            m.fit(dict["{}_train_X".format(d)], dict["{}_train_y".format(d)])
            lb_label1NN, pruning_power = m.predict(dict["{}_test_X".format(d)])
            
            lb_diff = lb_label - dict["{}_test_y".format(d)]
            lb_wrong_predict_count = np.count_nonzero(lb_diff)
            lb_acc = (dict["{}_test_y".format(d)].shape[0] - lb_wrong_predict_count) / dict["{}_test_y".format(d)].shape[0]
            tocc = time.time()
            dataset_result['1NN_{}{}_acc'.format(algo, WARPING_WINDOW)] = lb_acc
        
            dataset_result['1NN_{}{}_runtime'.format(algo, WARPING_WINDOW)] = tocc - ticc
            
            dataset_result['{}{} Speed Up'.format(algo, WARPING_WINDOW)] = dataset_result['1NN_{}{}_runtime'.format(algo, WARPING_WINDOW)] / dataset_result['Bounded1NN_{}{}_runtime'.format(algo, WARPING_WINDOW)]
            dataset_result['{}{} acc_diff'.format(algo, WARPING_WINDOW)] = dataset_result['Bounded1NN_{}{}_acc'.format(algo, WARPING_WINDOW)] - dataset_result['1NN_{}{}_acc'.format(algo, WARPING_WINDOW)]

    results['{}'.format(d)] = dataset_result

df = pd.DataFrame.from_dict(results, orient = 'index')
df.to_excel("MSMwindow_Exp_130.xlsx")

