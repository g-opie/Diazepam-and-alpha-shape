# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:04:47 2024

@author: a1147969

script cycles through ssd components, runs EMD, displays outcome for alpha band by plotting IMF relative to 
SSD timeseries, allowing visual inspection. Well fit data are then subject to cycle anlysis and extraction
of phase-aligned instantaneous frequency

"""

import os
import numpy as np
import emd
import pickle
import mne
import matplotlib.pyplot as plt

from base import ssd
from base.params import mne_datIn, srate, emd_mask_frq, ID, time, imf_pk

for sub in range(len(ID)):
    
    subOut = mne_datIn + ID[sub] 
    os.chdir(subOut)
    
    ssdDat = pickle.load(open('emd/cmp_dat.pkl', 'rb'))
    
    comp_kys = list(ssdDat.keys())

    emd_out = {}    
    
    # load filters
    ssd_out = pickle.load(open(f'ssd/{ID[sub]}_ssd_out.pkl', 'rb'))
    
    eeg_dat = {}

    for cmp_id, cmp in enumerate(comp_kys):
        
        kys_segs = comp_kys[cmp_id].split('_')
        
        cnd = kys_segs[0]
        frq = kys_segs[1]
        ssd_n = kys_segs[-1]
        
        if frq == 'a':        
            # get filter
            filt = ssd_out[f'{cnd}_{frq}_frq']['filt']
    
            imf_out = {}
            emd_dat = {}        
            
            # run emd on pre-post data
            for tm in ('PRE', 'POS'):
                
                # load eeg data
                raw = mne.io.read_raw_fif(f'pre_proc/{ID[sub]}_{cnd}{tm}_filtCh_raw.fif', preload=True)
                
                # get filtered data
                raw_filt = ssd.apply_filters(raw, filt, prefix="ssd")
                
                emd_in = raw_filt.get_data(picks=ssd_n)[0]        
                        
                #run EMD
                imf = emd.sift.mask_sift(emd_in,
                                          mask_amp = 2,
                                          # mask_amp_mode='ratio_sig',
                                          mask_freqs = np.array(emd_mask_frq[frq])/srate,                                   
                                          max_imfs = 6,
                                          envelope_opts = {"interp_method":"mono_pchip"})
                
                imf_out[tm] = imf
                emd_dat[tm] = emd_in
                
            # plot imfs against ssd time course
            for im_d in imf_out:
                
                fig, ax = plt.subplots()
                
                plt.ion()
                plt.plot(emd_dat[im_d])
                plt.plot(imf_out[im_d][:,imf_pk[frq]])
                plt.grid(visible=True, axis='x')
                plt.xlim(0, 1800)
                plt.title(f'freq = {frq}_{im_d}')
                
                plt.show(block=True)            
                                      
            # is the fit/data good to keep?
            keep = input('Keep component (Y/N)?')
            
            # if its good, extract cycle and paif info
            if keep == 'Y':        
                
                for n, im_d in enumerate(imf_out):
                    # calculate inst. phase/amp
                    IP, IF, IA = emd.spectra.frequency_transform(imf_out[im_d], srate, 'nht')
                        
                    # select good/thresholded cycles
                    amp_thr = np.percentile(IA[:, imf_pk[frq]], 75)	#cut-off set at 75th percentile
                    mask = IA[:, imf_pk[frq]] > amp_thr
                    cycles = emd.cycles.get_cycle_vector(IP[:, imf_pk[frq]], 
                                                          return_good=True, 
                                                          mask=mask)
                    
                    # phase align
                    pa_if, _ = emd.cycles.phase_align(IP[:, imf_pk[frq]],
                                                      IF[:, imf_pk[frq]],
                                                      cycles=cycles[:, 0])                
                
                    emd_out[f'{cnd}_{ssd_n}_{time[n]}'] = dict(imfs=imf_out[im_d],
                                                    IA=IA[:,imf_pk[frq]], 
                                                    IP=IP[:,imf_pk[frq]], 
                                                    IF=IF[:,imf_pk[frq]],
                                                    paif=pa_if,
                                                    mask=mask)   
                        
    pickle.dump(emd_out, open('emd/emd_out.pkl', 'wb'))
                
                
                
                
        