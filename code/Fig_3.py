# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:48:46 2025

@author: a1147969
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from string import ascii_uppercase as lts
import mne

from base.params import grp_dat_path, cond, mPath, colors

lobe_lbls = (('Temporal', 'Occipital', 'Sensorimotor', 'Parietal'), (0.81,0.64,0.46,0.3), (0.515,0.515,0.47,0.522))

# load position file for topoplots
pos = pickle.load(open(f'{mPath}working/pos.pkl', 'rb'))

# setup figure
mfig = plt.figure(figsize=(8,10), dpi=600)   
mfig.subplots_adjust(left=0, right=1, top=0.85, bottom=0.15)#, wspace=0.5, hspace=0.2)



subfigs = mfig.subfigures(5, 3,
                          width_ratios = [1,0.3,1],
                          height_ratios = [0.15,1,1,1,1])

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

# use first row of figs for condition labels
for lbls in zip(('Placebo', 'Diazepam'),(0,2)):
    subfigs[0,lbls[1]].suptitle(lbls[0], y=0.2, fontsize=24, fontweight=525, color='purple', fontstyle='italic')   

# plot IF and patterns
for lb_n, lb in enumerate(zip(('temp', 'occ', 'sm', 'par'),(0,1,2,4))):
    for cnd_n, cnd in enumerate(cond):
        
        if cnd == 'placebo':
            plt_ind = (1,0)
            fig_ind = 0
        else:
            plt_ind = (1,0)
            fig_ind = 2
            
        # add axes to subfig
        axs = subfigs[lb_n+1, fig_ind].subplots(1,2, gridspec_kw = {'wspace':0}) 
        
        # add panel label
        if cnd_n == 0:            
            axs[0].annotate(lts[lb_n], xy=(-0.55, 1.05), xycoords='axes fraction', fontsize=30, fontweight=200)
            
            
        # get pattern data
        patt = patterns[cnd][lb[0]]            
        patt_avg = np.mean(np.abs(np.vstack(patt)), axis=0)
        
        # get paif data
        paif_pre = paif_df_demean[(paif_df_demean['cond']==cnd_n) & 
                                  (paif_df_demean['lobe']==lb[1]) & 
                                  (paif_df_demean['good_trials'] == True) & 
                                  (paif_df_demean['time'] == 0)]

        paif_pos = paif_df_demean[(paif_df_demean['cond']==cnd_n) & 
                                  (paif_df_demean['lobe']==lb[1]) & 
                                  (paif_df_demean['good_trials'] == True) & 
                                  (paif_df_demean['time'] == 1)]
        
        for df in (paif_pre, paif_pos):
            df.drop(['sub', 'time','cond', 'lobe', 'cycle', 'ssd', 'good_trials'], axis=1, inplace=True)            
            
        # plot avg pattern
        img, _ = mne.viz.plot_topomap(patt_avg, 
                                      pos, 
                                      axes=axs[plt_ind[0]],
                                      sensors=False, 
                                      show=False,
                                      res=600)
        

        # plot paif
        for vl in (12, 36):            # ref lines for peak/trough
            axs[plt_ind[1]].axvline(x=vl, linestyle = ':', color='grey', linewidth=1)
            
        for pf_n, pf in enumerate(zip((paif_pre, paif_pos), ('Pre', 'Post'))):
            axs[plt_ind[1]].plot(pf[0].mean(axis=0), label=pf[1], color=colors[pf_n], linewidth=3)
            
        # formatting
        axs[plt_ind[1]].set_xticks(np.arange(5)*12, fontsize=12, labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])                
        axs[plt_ind[1]].set_ylabel('Inst. Freq (Hz)', fontsize=12)
        axs[plt_ind[1]].spines[['right', 'top']].set_visible(False)

            
        if cnd_n == 0:
            axs[plt_ind[1]].annotate(lobe_lbls[0][lb_n], 
                                     xy=(-0.5, 0.5), 
                                     xycoords='axes fraction', 
                                     fontsize=15, 
                                     fontweight=535,
                                     color='mediumslateblue',
                                     rotation=90,
                                     va='center')
        
        # lobe specific formatting
        if lb_n == 3 and cnd_n == 1:
            axs[plt_ind[1]].legend(loc=(0.6,0.75), frameon=False)
        
        if lb_n == 3:
            axs[plt_ind[1]].set_xlabel('Phase', fontsize=15, fontstyle='italic')
            
        for ax in axs.flat:
            fmt = '%.1f'  
            yticks = mtick.FormatStrFormatter(fmt)
            ax.yaxis.set_major_formatter(yticks)
            
# save fig
mfig.savefig(f'{mPath}Fig 3.jpg', dpi=600, bbox_inches='tight')