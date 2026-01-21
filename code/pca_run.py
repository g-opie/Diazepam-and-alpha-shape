# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:54:14 2025

@author: a1147969

load paif data, run PCA, format scores for stats

"""

import os
import sails
import pickle
import numpy as np
import pandas as pd

from base.params import time, cond, grp_dat_path

os.chdir(f'{grp_dat_path}emd/')

paif_in = pickle.load(open('paif.pkl', 'rb'))

paif_out = []

for c_n, cnd in enumerate(cond):
    for t_n, tme in enumerate(time):
        for x in paif_in[cnd][tme]:
            paif_out.append(x)
        
paif_all = np.concatenate(paif_out)

# get dataset without dummy codes for PCA
pc_dat_in = np.delete(paif_all, np.s_[48:], 1).astype(float)

# demean data for PCA
cyc_mean = np.mean(pc_dat_in, axis=1)[:,None]
phase_mean = np.mean(pc_dat_in, axis=0)[:,None]

pc_dat_in = pc_dat_in - cyc_mean

# check for bad trials
bads, _ = sails.utils.gesd(pc_dat_in.std(axis=1))
goods = bads == False

# append good trial index to paif df, save
idx = np.reshape(goods, (goods.shape[0],1))

paif_df = pd.DataFrame(np.append(paif_all, idx, axis=1), columns=np.hstack((np.arange(0,48,1), 'sub','time','cond','lobe', 'cycle','ssd','good_trials')))

paif_df.to_csv('paif_df.csv', index=False)
    
# run PCA
pca = sails.utils.PCA(pc_dat_in[goods, :], npcs=10)

# save pca object
pickle.dump(pca, open('pca_obj.pkl', 'wb'))

# format PC scores for stats
paif_df_sub = paif_df[paif_df['good_trials'] == 'True']

pcs_df = pd.DataFrame(np.append(pca.scores, paif_df_sub.iloc[:,-7:], axis=1), columns=np.hstack((['PC'+str(x+1) for x in range(10)], 'sub','time','cond','lobe', 'cycle','ssd','good_trials')))

pcs_df.to_csv('pc_scores_df.csv', index=False)
    
