# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:52:54 2025

@author: a1147969
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sb

from specparam.data.periodic import get_band_peak_group
from specparam.plts.periodic import plot_peak_fits
from specparam.plts.aperiodic import plot_aperiodic_fits 

from base.params import cond, time, grp_dat_path, mPath, cnd_lab, colors        

lab = ('Pre', 'Post')

# formatting
p_dict = {'Log Power':((0, 1.8),
                   (0, 1.15),
                   (0, 1.15)),
          
          'Freq':((6.5, 14.5),
                  (14.5, 21),
                  (20, 30.5)),
          
          'Peak':((0, 1.7),
                  (0, 1.1),
                  (0, 1)),
          
          'Freq_ytick':((8, 10, 12, 14),
                        (16, 18, 20),
                        (20, 25, 30))          
          }

#setup figure 
fig = plt.figure(figsize=(8.5,9), dpi=600)   
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)#, wspace=0.5, hspace=0.2)                        

subfigs = fig.subfigures(3, 1, #height_ratios = [1.2,1,1],
                           wspace = 0.1,
                           hspace = 0.4)                         

spt_axs = subfigs[0].subplots(1,2)
ap_axs =  subfigs[1].subplots(1,5, width_ratios=[0.55,0.15,0.007, 0.55,0.15], gridspec_kw={'wspace':0.25})
al_axs =   subfigs[2].subplots(1,7, width_ratios=[0.6,0.15,0.15,0.007, 0.6,0.15,0.15], gridspec_kw={'wspace':0.4})

# add panel labels
for fx_id, fx in enumerate((zip(('A', 'B', 'C'), (0, 1, 2)))):    
    subfigs[fx[1]].suptitle(fx[0], x=0.025, y=1.2, fontsize=26, fontweight='medium')
    
# remove dummy axes
for ax in (ap_axs[2], al_axs[3]):  
    ax.set_axis_off()

# load data
os.chdir(f'{grp_dat_path}/spect/')

ap = pickle.load(open('ap_ROI.pkl', 'rb'))
spect = pickle.load(open('spect_ROI.pkl', 'rb'))
alpha = pickle.load(open('alpha_ROI.pkl', 'rb'))

freqs  = np.load(f'{mPath}working/freq.npy')

for cnd_id, cnd in enumerate(cond):
    for tm_id, tm in enumerate(time):
        
#### spectra ####
        if cnd_id == 0:
            
            ap_id = 0
            p_id = 0
        else:
            
            ap_id = 3
            p_id = 4

        dat = np.mean(spect[f'{cnd}_{tm}'], axis=0)
        err = np.std(spect[f'{cnd}_{tm}'], axis=0)/np.sqrt(len(spect[f'{cnd}_{tm}']))
        
        spt_axs[cnd_id].plot(freqs, dat, label=lab[tm_id], linewidth=1.5, color=colors[tm_id])
        spt_axs[cnd_id].fill_between(freqs, dat-err, dat+err, alpha=0.3, color=colors[tm_id])
        spt_axs[cnd_id].set_ylim(bottom=-10.6, top=-8.5)
        
        # formatting                
        if tm_id == 1:
            spt_axs[cnd_id].set_xlabel('Frequency', fontsize=14)
            spt_axs[cnd_id].spines[['top', 'right']].set_visible(False) 
            spt_axs[cnd_id].legend(loc='lower left', frameon=False)
            spt_axs[cnd_id].annotate(cnd_lab[cnd_id], xy=(0.9, 1.1), xycoords='axes fraction', fontsize=24, fontweight=525, ha='right', color='purple', fontstyle='italic')
        
#### aperiodic params ####
    # plot ap fits across subjects
    plot_aperiodic_fits([np.vstack(ap[f'{cnd}_PRE']), np.vstack(ap[f'{cnd}_POS'])], 
                        [2, 40],
                        ax = ap_axs[ap_id],
                        colors=colors)
    
    ap_axs[ap_id].set_ylim(bottom=-11, top=-7.8)
    
    # plot ind subj exponent    
    pre = np.vstack(ap[f'{cnd}_PRE'])[:,1]
    pos = np.vstack(ap[f'{cnd}_POS'])[:,1]
    
    ap_pal = {0:colors[0], 1:colors[1]}
    
    ap_ln = sb.lineplot(data = [np.median(pre), np.median(pos)], ax=ap_axs[ap_id+1], c='darkgrey')    
    ap_bx = sb.boxplot(data = [pre, pos], ax=ap_axs[ap_id+1], width=0.5, palette=ap_pal)
    
    # format
    ap_axs[ap_id+1].spines[['right', 'top']].set_visible(False)
    ap_axs[ap_id+1].set_xticklabels(('Pre', 'Pos'))
    ap_axs[ap_id+1].set_ylim(bottom=0.4, top=2.2)
    
    # add sig marker (based on outputs of glmms)
    if cnd_id == 1:
        ap_axs[ap_id+1].annotate('*', xy=(1.5, np.median(pos)), fontsize=12, fontweight='semibold', ha='center', va='top')
        
#### periodic params ####
    # load group fits
    p_pre = pickle.load(open(f'{grp_dat_path}/spect/{cnd}_PRE_grp_spect_fit_ROI.pkl', 'rb'))
    p_pos = pickle.load(open(f'{grp_dat_path}/spect/{cnd}_POS_grp_spect_fit_ROI.pkl', 'rb'))    
          
    # get periodic fits
    x1_f = get_band_peak_group(p_pre, [7,14])
    x1_f_na = np.nan_to_num(x1_f, nan=np.nanmean(x1_f, axis=0))
            
    x2_f = get_band_peak_group(p_pos, [7,14])
    x2_f_na = np.nan_to_num(x2_f, nan=np.nanmean(x2_f, axis=0))
            
    # plot periodic fits across subjects
    plot_peak_fits([x1_f_na, x2_f_na], average='mean', colors = colors, ax = al_axs[p_id])
    
    # format periodic fits
    al_axs[p_id].set_ylim(bottom = p_dict['Peak'][0][0], top = p_dict['Peak'][0][1])
    
    # plot ind subj freq  
    sb.lineplot(data = [np.median(x1_f_na[:,0]), np.median(x2_f_na[:,0])], ax=al_axs[p_id+1], c='darkgrey')
    sb.boxplot(data = [x1_f_na[:,0], x2_f_na[:,0]], ax=al_axs[p_id+1], width=0.5, palette=ap_pal)
    
    # plot ind subj power        
    sb.lineplot(data = [np.median(x1_f_na[:,1]), np.median(x2_f_na[:,1])], ax=al_axs[p_id+2], c='darkgrey')
    sb.boxplot(data = [x1_f_na[:,1], x2_f_na[:,1]], ax=al_axs[p_id+2], width=0.5, palette=ap_pal)
    
    # annotate sig, based on glmms
    if cnd_id == 1:
        al_axs[p_id+2].annotate('*', xy=(1.5, np.nanmedian(x2_f[:,1], axis=0)), fontsize=12, fontweight='semibold', ha='center', va='top')
    
    # format
    for ind, p_ax in enumerate(zip((al_axs[p_id+1], al_axs[p_id+2]), ('Freq', 'Log Power'))):
        p_ax[0].spines[['right', 'top']].set_visible(False)
        p_ax[0].set_xticklabels(('Pre', 'Post'))
        p_ax[0].xaxis.set_tick_params(labelrotation=45)            
        p_ax[0].set_ylabel(p_ax[1], ha='left', y=1.05, rotation=0, labelpad=0, fontstyle='italic')
        p_ax[0].set_ylim(bottom=p_dict[p_ax[1]][0][0], top=p_dict[p_ax[1]][0][1])
        
        if ind == 0:
            p_ax[0].set_yticks((p_dict['Freq_ytick'][0]))
        else:
            p_ax[0].set_yticks((0, 1))
            
    for ax in (spt_axs[cnd_id], ap_axs[ap_id], al_axs[p_id]):       #, bL_axs[p_id], bU_axs[p_id]): 
        ax.set_ylabel('Log Power', ha='left', y=1.05, fontsize=13, rotation=0, labelpad=0, fontstyle='italic')
        ax.spines[['left','bottom']].set(linewidth=1)
        
# save fig
fig.savefig(f'{mPath}Fig_2.jpg', dpi=600, bbox_inches='tight')
        


    
    
        
        
        
        
        
            
            
                
            
            
            
        

            
            
            
            
            
            
        
        
        