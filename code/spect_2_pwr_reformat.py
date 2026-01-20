# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:52:54 2025

@author: a1147969
"""

import os
import pickle


from specparam.bands import Bands
from specparam.data.periodic import get_band_peak

from params import ID, cond, time, mne_datIn, grp_dat_path

grp_spect = {'placebo_PRE':[], 'placebo_POS':[], 'diazepam_PRE':[], 'diazepam_POS':[]}
grp_alpha = {'placebo_PRE':[], 'placebo_POS':[], 'diazepam_PRE':[], 'diazepam_POS':[]}
grp_beta_L = {'placebo_PRE':[], 'placebo_POS':[], 'diazepam_PRE':[], 'diazepam_POS':[]}
grp_beta_U = {'placebo_PRE':[], 'placebo_POS':[], 'diazepam_PRE':[], 'diazepam_POS':[]}
grp_ap = {'placebo_PRE':[], 'placebo_POS':[], 'diazepam_PRE':[], 'diazepam_POS':[]}

#Define bands
bands = Bands({'alpha': [7, 14],
               'beta_L': [15, 20],
               'beta_U': [21, 30]})

# get ind subj data
for sub_id, sub in enumerate(ID):
    os.chdir(f'{mne_datIn}/{sub}/spect/')
    
    fm = pickle.load(open(f'{sub}_spect_fits_ROI.pkl', 'rb'))

    for cnd_id, cnd in enumerate(cond):
        for tm_id, tm in enumerate(time):
            
            dat = fm[f'{cnd}_{tm}']
            
            grp_spect[f'{cnd}_{tm}'].append(dat.power_spectrum)
            grp_ap[f'{cnd}_{tm}'].append(dat.aperiodic_params_)
            grp_alpha[f'{cnd}_{tm}'].append(get_band_peak(dat, bands['alpha']))
            grp_beta_L[f'{cnd}_{tm}'].append(get_band_peak(dat, bands['beta_L']))
            grp_beta_U[f'{cnd}_{tm}'].append(get_band_peak(dat, bands['beta_U']))
            
# save dat
for fl in zip((grp_spect, grp_ap, grp_alpha, grp_beta_L, grp_beta_U),('spect','ap','alpha','beta_L','beta_U')):
    pickle.dump(fl[0], open(f'{grp_dat_path}/{fl[1]}_ROI.pkl', 'wb'))
    

            
            
        
    
            
                
                
            
            
            
        

            
            
            
            
            
            
        
        
        