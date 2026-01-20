# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:10:53 2025

@author: a1147969

script restructures paif data to suit stats and PCA

"""

import os
import pickle
import pandas as pd

from params import mne_datIn, ID, time, grp_dat_path
from base.helper import get_dummy_codes


paif = {'placebo': {'PRE':[], 'POS':[]},
        'diazepam': {'PRE':[], 'POS':[]}}

cnd_dict = {'temp':0, 'occ':1, 'sm':2, 'fr':3, 'par':4}

for sub in range(len(ID)):
    
    subOut = mne_datIn + ID[sub] 
    os.chdir(subOut)
    
    emdDat = pickle.load(open('emd/emd_out.pkl', 'rb'))
    
    if bool(emdDat):    
        srcs = {'placebo': pd.read_csv(f'{mne_datIn}{ID[sub]}/sources/{ID[sub]}_placebo_a_frq_srcs.csv'),
                'diazepam': pd.read_csv(f'{mne_datIn}{ID[sub]}/sources/{ID[sub]}_diazepam_a_frq_srcs.csv')}
        
        kys = list(emdDat.keys())
        
        # get list of components
        comp_list = [kys[x] for x in range(len(kys)) if 'PRE' in kys[x]]
        
        for cmp in comp_list:
            
            # get condition and comp number for each
            kys_segs = cmp.split('_')
            
            cnd = kys_segs[0]
            ssd_n = kys_segs[1]
    
            # get some source info for component            
            src = srcs[cnd][srcs[cnd]['Unnamed: 0'] == int(ssd_n[-1])]
            
            # loop pre-post time, add paif data to group array, dummy-coded for conditions            
            for t_n, t in enumerate(time):
                
                paif[cnd][t].append(get_dummy_codes(emdDat[f'{cnd}_{ssd_n}_{t}']['paif'].T, 
                                                    sub=ID[sub],
                                                    time=t_n,
                                                    cnd=cnd,
                                                    lbe=cnd_dict[src.iloc[0,-1]],
                                                    ssd=ssd_n))
            
# save paif dict
pickle.dump(paif, open(f'{grp_dat_path}emd/paif.pkl', 'wb'))



            
            
                

            

                
            
        
        