# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:06:52 2024

@author: a1147969
"""
import numpy as np
import os
import mne
import pandas as pd
import pickle
from specparam.bands import Bands

from base.params import mPath, mne_datIn, cond, time, ID
from base import ssd, helper


def run_ssd(raw, peak, s_bp = 2, n_bp = 4, n_bs = 3):
    
    #set ssd vars
    signal_bp = [peak - s_bp, peak + s_bp]
    noise_bp = [peak - n_bp, peak + n_bp]
    noise_bs = [peak - n_bs, peak + n_bs]
    
    #run ssd
    filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)
    
    return filters, patterns

    
#Define bands for SSD
bands = Bands({'a_frq': [7, 14],
               'lb_frq': [15, 20],
               'ub_frq': [21, 30]})

# load group spect data
frq_dat = pd.read_csv(f'{mPath}stats/spect/data/spect_dat.csv')

#loop subject
for subN in range(len(ID)):
    
    subOut = mne_datIn + ID[subN] + '/'    
    os.chdir(subOut)
    
    # dict for ssd outputs
    ssd_out = {}
    
    for cnd in range(len(cond)):            
        
        # get subject spect data
        sub_spect = frq_dat[(frq_dat['Subj'] == subN+1) & (frq_dat['Grp'] == cnd)]
        
        raws = []
        
        for tme in range(len(time)):
            
            #load data
            raw = helper.load_raw_eeg(ID[subN], cond[cnd], time[tme])
            raws.append(raw)                
            
        #concatenate over time, within condition, prior to ssd
        con_raw = mne.concatenate_raws(raws=raws)
        
        # run ssd for each band of interest
        for n, (bnd_lab, bnd) in enumerate(bands):
            
            # get target frq, avgd over time
            peak_frq = sub_spect[bnd_lab].mean()
            
            if np.isnan(peak_frq):
                
                # skip band if no peak available (pre or post)
                filt, patt = np.nan, np.nan
            else:
                
                # run ssd
                filt, patt = run_ssd(con_raw, peak_frq)
            
            ssd_out[f'{cond[cnd]}_{bnd_lab}'] = dict(filt = filt, pattern = patt, peak = peak_frq)
        
    if not os.path.isdir(f'{mne_datIn}/{ID[subN]}/ssd/'):
        os.makedirs(f'{mne_datIn}/{ID[subN]}/ssd/')

    #save outputs        
    pickle.dump(ssd_out, open(f'ssd/{ID[subN]}_ssd_out.pkl', 'wb'))
    
    
    