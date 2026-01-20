# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:08:05 2024

@author: a1147969
"""

import os
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from specparam import SpectralModel

from ..base.params import ID, cond, time, mne_datIn

#extract subj specific spect data, produce individual plots of specparam fit and oscillation topography
for sub in range(len(ID)):
    dat_d = f'{mne_datIn}/{ID[sub]}/ICA/'            
    os.chdir(dat_d)                
    fm_out = {}
    
    #setup figure
    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(16, 10), dpi=600)
    
    for cnd in range(len(cond)):
        for tm in range(len(time)):
            #specparam for each timepoint/condition
            #load data
            raw = mne.io.read_raw_fif(f'{dat_d}/{ID[sub]}_{cond[cnd]}_{time[tm]}_filtCh_IC_raw.fif', preload=True)            
            
            # avg ref
            raw_avg_ref = raw.set_eeg_reference(ref_channels='average')
            
            # psd (epochs)            
            raw_events = mne.make_fixed_length_events(raw, duration=10)
            
            ep = mne.Epochs(raw, raw_events, tmin=0, tmax=10, baseline=None)
            
            ep_psd = ep.compute_psd(method='multitaper', fmin = 2, fmax = 40, tmin=0, tmax=10).get_data(return_freqs=True) #n_fft=2000, n_overlap=1000
            
            #Initialize SpectralModel object
            fm = SpectralModel(peak_width_limits=[1, 12],
                               max_n_peaks=6, 
                               min_peak_height=0.1,
                               peak_threshold=1.2,
                               verbose=False)

            # get channels (C1 [32], Cz [18], C2 [33], CP1 [22], CPz [59], CP2 [23], Pz [19])
            ch_dat = np.mean(ep_psd[0][:, [18, 19, 22, 23, 32, 33, 59], :], axis=0)
            ch_dat = np.mean(ch_dat, axis=0)
            
            #Define frequency range, fit model, plot results
            freq_range = [2, 40]
            fm.report(ep_psd[1], ch_dat, freq_range, ax=axs[cnd, tm])
            
            axs[cnd, tm].set_title(f'{cond[cnd]}_{time[tm]}')
            axs[cnd, tm].annotate(f'R2 = {round(fm.r_squared_,3)}, err = {round(fm.error_,3)}', xy=(0.05,0.1), fontsize=15, xycoords='axes fraction')
            
            fm_out[f'{cond[cnd]}_{time[tm]}'] = fm            
    
    fig.suptitle(f'{cond[cnd]}_{time[tm]}', fontsize=25)    
        
    os.chdir(f'{mne_datIn}/{ID[sub]}/spect/')
            
    # save fm list
    pickle.dump(fm_out, open(f'{ID[sub]}_spect_fits_ROI.pkl', 'wb'))
    
    fig.suptitle('ROI')
                
    #save fit check fig
    fig.savefig(f'{ID[sub]}_ROI_fitCheck.jpg')
    
    
    
    
    
    
    
    
    
    
    