# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:19:37 2023

@author: a1147969
"""

import numpy as np
import os
import emd
import pickle
import matplotlib.pyplot as plt
from string import ascii_uppercase as lts
from matplotlib import ticker as mtick
import pandas as pd


from base.params import grp_dat_path, mPath

# load position file for topoplots
pos = pickle.load(open(f'{mPath}pos.pkl', 'rb'))

os.chdir(grp_dat_path)

#number of PCs to plot
numPCs = 4

#dict for storing norm waves
norm_waves = {}
prctl_if_prof = {}
pcaObj_all = []

# ---------------- plot PCs ---------------- # 

# change dir
os.chdir(grp_dat_path)

# load patterns
patterns = pickle.load(open('ssd/grp_patterns.pkl', 'rb'))

# load paif data
paif_df = pd.read_csv('emd/paif_df.csv')

# remove cycle averages
pf_np = np.array(paif_df.iloc[:,:48])

cycle_mean = np.mean(pf_np, axis=1)[:,None]
pf_np = pf_np - cycle_mean

paif_df_demean = paif_df.copy(deep=True)
paif_df_demean.iloc[:,:48] = pf_np

# add subplots
fig, pc_axs = plt.subplots(2, 4, figsize=(10,6), dpi=600, gridspec_kw={'wspace':0.3})
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)#, wspace=0.5, hspace=0.2)

# load pca object
pca = pickle.load(open('emd/pca_obj.pkl', 'rb'))

# load paif data
paif = paif_df[paif_df['good_trials'] == True]

phase_avg = np.array(paif.iloc[:,:48].mean(axis=0))

scores = pca.scores    

# get PC scores for each percentile, generate norm wave for each
for i in range(numPCs):
    percVals = np.linspace(np.percentile(scores[:,i],0,axis=0), np.percentile(scores[:,i],100,axis=0), 20) 
    
    for ii in range(percVals.shape[0]):
        pct_name = 'PC' + str(i+1) + '_' + str(ii)            
    
        proj_sc_dm = pca.project_score(percVals[ii])
        proj_sc_incM = proj_sc_dm + phase_avg.T
        
        #store PAIF profile for each percentile
        prctl_if_prof[pct_name] = proj_sc_dm[i,:]   #proj_sc_incM[i,:] 
                    
        #calc norm wave
        temp = emd.cycles.normalised_waveform(proj_sc_incM[i,:])
        norm_waves[pct_name] = temp[0]
            
#plot
col = plt.cm.winter(np.linspace(0,1, num=percVals.shape[0]))

nm_col = 2
if_col = 1
        
for i in range(numPCs):
    # ref lines for peak/trough
    for vl in (12, 36):            
        pc_axs[0,i].axvline(x=vl, linestyle = ':', color='grey', linewidth=1)        
        
    for ii in range(percVals.shape[0]):
        dat_idx = 'PC' + str(i+1) + '_' + str(ii)
        
        pc_axs[1, i].plot(norm_waves[dat_idx], c=col[ii], linewidth = 1)
        pc_axs[0, i].plot(prctl_if_prof[dat_idx], c=col[ii], linewidth = 1)
        
    #format panels
    if i == 0:            
        pc_axs[1, i].set_ylabel('Normalised waveform (au)', fontsize=15)
        pc_axs[0, i].set_ylabel('Inst. Freq (Hz)', fontsize=15)
        
    pc_axs[1, i].set_xticks(np.linspace(0,48,3),np.linspace(0,1,3))                
    pc_axs[0, i].set_xticks(np.arange(5)*12, labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])                           
        
    pc_axs[1, i].set_xlabel('Proportion of Cycle', fontsize=15)
    pc_axs[0, i].set_xlabel('Phase', fontsize=15)
    
    pc_axs[0, i].set_title(f'PC{i+1} ({np.round(pca.explained_variance_ratio[i]*100, 1)}%)', fontsize=18, fontweight=500)
    
    if i > 0:
        for ind in range(2):
            pc_axs[ind, i].set_ylabel('')
        
    
    ls = lts[i]
    column = 0    

for ax in pc_axs.flat:
    fmt = '%.1f'  
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.spines[['right', 'top']].set_visible(False)

os.chdir(mPath)
fname = 'Fig 4.jpg'    
plt.savefig(fname, dpi=600, bbox_inches='tight')  

            
            
            
   
    
    
    
    
            
            
            
            
            
    
            
    


    