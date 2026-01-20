# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:34:06 2025

@author: a1147969

visually inspect components timeseries

"""

import os
import pickle
import mne
import pandas as pd
from specparam.bands import Bands

from base.params import mne_datIn, cond, ID
from base import ssd 

#Define bands for SSD
bands = Bands({'a_frq': [7, 14],
               'lb_frq': [15, 20],
               'ub_frq': [21, 30]})

for sub in range(3, len(ID)):
    
    subOut = mne_datIn + ID[sub]   
    os.chdir(subOut)

    # load filters
    ssd_out = pickle.load(open(f'ssd/{ID[sub]}_ssd_out.pkl', 'rb'))

    # dict for exporting comp timecourse
    cmp_out = {}    
    
    for cnd_id, cnd in enumerate(cond):
        
        # load eeg data
        raw = mne.io.read_raw_fif(f'pre_proc/{ID[sub]}_{cnd}_filtCh_concat_raw.fif', preload=True)
        
        for bnd_n, (bnd_nm, bnd_rng) in enumerate(bands):            
            try:
                # load identified ssd comps (from get_sources)
                comps = pd.read_csv(f'sources/{ID[sub]}_{cnd}_{bnd_nm}_srcs.csv')
                
                if len(comps) == 0:
                    continue
                
                # get filter for band
                filt = ssd_out[f'{cnd}_{bnd_nm}']['filt']
                
                # get list of components to remove
                ssd_drop = ['ssd'+str(x) for x in range(filt.shape[1]) if x not in comps['Unnamed: 0'].values]

                # display component timecourses
                print(f'Showing timecourse data for {cnd}-{bnd_nm} of subject {ID[sub]}')
                raw_filt = ssd.apply_filters(raw, filt, prefix="ssd", drop_comp = ssd_drop)
                raw_filt.plot(block=True)
                
                # retrieve user input on components to keep
                comp_keep = input('Enter components to keep:')
                
                if comp_keep == "N":
                    continue
                else:
                    comp_keep = [int(s) for s in comp_keep.split(' ')]
                    
                # get comp timecourses
                cmp_dat = raw_filt.get_data()
                
                # export selected component timecourse for EMD
                for cmp_id, cmp in enumerate(comp_keep):
                    
                    # get index of ssd
                    idx = [x for x in range(len(raw_filt.ch_names)) if raw_filt.ch_names[x] == 'ssd'+str(cmp)]
                    
                    # get relevant comp data
                    cmp_out[f'{cnd}_{bnd_nm}_ssd{str(cmp)}'] = cmp_dat[idx]
                    
                
            except FileNotFoundError:
                print('{} not found, moving to next band'.format(bnd_nm))
                
    # save
    if not os.path.isdir(f'{mne_datIn}/{ID[sub]}/emd/'):
        os.makedirs(f'{mne_datIn}/{ID[sub]}/emd/')
        
    pickle.dump(cmp_out, open(f'{mne_datIn}{ID[sub]}/emd/cmp_dat.pkl', 'wb'))
                
                
                
                
                    
                

            
