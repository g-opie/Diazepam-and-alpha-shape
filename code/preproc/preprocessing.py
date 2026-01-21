# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:35:53 2024

@author: a1147969
"""

import os
import mne
from mne.preprocessing import read_ica
from pyprep.find_noisy_channels import NoisyChannels

from base.params import eeglab_datIn, mne_datIn, cond, time, ID


#%% load eeglab data, bandpass filt, id bad ch & interpolate
for subN in range(len(ID)):
    
    #check for sub dir, create if not present
    subOut = mne_datIn + ID[subN] + '/'
    if not os.path.isdir(subOut):
        os.mkdir(subOut)
    
    os.chdir(subOut)    
    
    for cnd in range(len(cond)):
        raws = []
        badCh = []
        for tm in range(len(time)):
            fname = ID[subN] + '_' + cond[cnd] + time[tm]            
            
            #load file
            eLab_fname = eeglab_datIn + ID[subN] + '/' + fname + '_line.set'
            raw = mne.io.read_raw_eeglab(eLab_fname, preload=True, montage_units='dm')
            
            #mod montage
            raw.rename_channels(mapping={'Afz':'AFz'})
            raw.set_montage('standard_1005')
            
            #filter
            raw.filter(1, 100)
            
            #find bad ch
            bad = NoisyChannels(raw)
            
            bad.find_all_bads(channel_wise=True)
            
            badCh.append(bad.get_bads())
            
            #id bad data
            raw.plot(block=True)               
            
            raws.append(raw)
        
        #group bad channels
        com_bads = list(set(badCh[0] + badCh[1]))
        
        for x in range(len(raws)):
            raws[x].info['bads'] = com_bads
            
            raws[x].interpolate_bads()
            
            raws[x].save(ID[subN] + '_' + cond[cnd] + time[x] + '_filtCh_raw.fif', overwrite=True)

#%% train ICA
from mne.preprocessing import ICA
for subN in range(len(ID)):    
    
    subOut = mne_datIn + ID[subN] + '/'    
    os.chdir(subOut)    
    
    for cnd in range(1):#len(cond)):
        raws = []        
        for tm in range(len(time)):
            fname = ID[subN] + '_' + cond[cnd] + time[tm] + '_filtCh_raw.fif'
            
            raw = mne.io.read_raw_fif(fname, preload=True)
            
            raws.append(raw)
            
        #concatenate prior to ica
        con_raw = mne.concatenate_raws(raws=raws)     
                
        #ICA training
        ica = ICA(n_components=15)
        icFit = ica.fit(con_raw)
        
        #save ica object
        icFit.save(ID[subN] + '_' + cond[cnd] + '_ica.fif', overwrite=True)

  #%% remove ICs        
for subN in range(len(ID)):
    
    subOut = mne_datIn + ID[subN] + '/'    
    os.chdir(subOut)    
    
    for cnd in range(1):#len(cond)):
        
        #load ICA object
        icObj = read_ica(ID[subN] + '_' + cond[cnd] + '_ica.fif')
        
        #load raw data, concat for IC removal
        raws = []
        for tm in range(len(time)):
            #load file
            fname = ID[subN] + '_' + cond[cnd] + time[tm] + '_filtCh_raw.fif' 
            raw = mne.io.read_raw_fif(fname, preload=True)
            
            raws.append(raw)        
        
        con_raw = mne.concatenate_raws(raws=raws)
        
        icObj.plot_sources(con_raw, block=True)
        icObj.save(ID[subN] + '_' + cond[cnd] + '_ica.fif', overwrite=True)
        
        #save cleaned version of data
        for tm in range(len(time)):
            #load file
            fname = ID[subN] + '_' + cond[cnd] + time[tm] + '_filtCh_raw.fif' 
            raw = mne.io.read_raw_fif(fname, preload=True)            
            
            icObj.apply(raw)
            
            raw.save(ID[subN] + '_' + cond[cnd] + '_' + time[tm] + '_filtCh_IC_raw.fif', overwrite=True)   

