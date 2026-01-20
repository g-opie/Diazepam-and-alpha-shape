# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:48:06 2024

@author: a1147969

get ssd spect characteristics, plot spatial patterns.

"""

from specparam.analysis import get_band_peak_group
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import mne
import pandas as pd

from base.params import mne_datIn, cond, time, ID
from base import ssd, helper

for subN in range(len(ID)):
    
    subOut = mne_datIn + ID[subN] + '/'    
    os.chdir(subOut)
    
    for cnd in range(len(cond)):
        
        #load filter
        ssd_out = pickle.load(open(ID[subN] + '_' + cond[cnd] + '_ssd_out.pkl', 'rb'))
        filters = ssd_out['filters']
        patterns = ssd_out['patterns']
        
        #load eeg
        fname = ID[subN] + '_' + cond[cnd] + time[0] + '_filtCh_raw.fif'
        raw = mne.io.read_raw_fif(fname, preload=True)
        
        # Save patterns and filters
        df = pd.DataFrame(filters.T, columns=raw.ch_names)
        df.to_csv(ID[subN] + '_' + cond[cnd] + '_filt.csv', index=False)
        df = pd.DataFrame(patterns.T, columns=raw.ch_names)
        df.to_csv(ID[subN] + '_' + cond[cnd] + '_patterns.csv', index=False)
        
        raw_trm = helper.reject_bad_segs(raw)                     
        
        #apply filters
        raw_ssd = ssd.apply_filters(raw_trm, filters)        
        
        SNR, fm = helper.get_SNR(raw_ssd)
        
        bands = pd.DataFrame(get_band_peak_group(fm, [7, 14]),columns=['Freq.', 'PW', 'BW'])    #.dropna()
        bands['SNR'] = SNR
        bands.to_csv(subOut + cond[cnd] + '_ssdComp_foof_out.csv')
        
        bands = bands.dropna()
        
        row = round(len(bands)/4 + 0.5)
        
        fig, axs = plt.subplots(nrows=row, ncols=4, figsize=(15,15), dpi=600)        
                        
        #plot            
        for i, ax in enumerate(axs.flat):                               
            if i >= len(bands):                
                ax.spines[['left','right','top','bottom']].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            p = patterns[:,bands.index[i]]
            idx = np.argmax(np.abs(p))
            sign = np.sign(p[idx])
            mne.viz.plot_topomap(sign * patterns[:, bands.index[i]], raw.info, axes=ax, res=600, show=False)
            ax.set_title('ssd:{}, Fq:{}, SNR:{}'.format(bands.index[i], round(bands.iloc[i, 0],2),round(SNR[bands.index[i]],2)))
        
        figName = ID[subN] + '_' + cond[cnd] + '_ssdPatt_filt.jpg'
        plt.savefig(figName, dpi=400)   
        