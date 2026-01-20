# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:01:34 2025

@author: a1147969
"""
import emd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, cramervonmises_2samp
from string import ascii_uppercase as lts

from base.params import  mPath, colors, ex_wave_loc_dict, grp_dat_path   

# load pc data
os.chdir(f'{grp_dat_path}emd/')
pc_df = pd.read_csv('pc_scores_df.csv')
pc_df = pc_df[pc_df['good_trials'] == True]

pca = pickle.load(open('pca_obj.pkl', 'rb'))

# load paif data
pf_df = pd.read_csv('paif_df.csv')
pf_df = pf_df[pf_df['good_trials'] == True]

pf_pre = pf_df[(pf_df['time'] == 0) & (pf_df['cond'] == 1)]
pf_pos = pf_df[(pf_df['time'] == 1) & (pf_df['cond'] == 1)]

pre_ph_avg = np.array(pf_df.iloc[:,:48].mean(axis=0))
pos_ph_avg = np.array(pf_df.iloc[:,:48].mean(axis=0))

cnd_n, cnd = 1, 'diazepam'
    
# create plot, set details
fig = plt.figure(figsize=(7,8), dpi=600)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.2, hspace=0.3)

subfigs = fig.subfigures(3,1, height_ratios=[0.8,0.3,1])

# subfigs for distplots
dist_axs = subfigs[0].subplots(1,4)

# axes for example waves
wv_axs = subfigs[2].subplots(1,2)
  
y = {'min':[], 'max':[]}
x = {'min':[], 'max':[]}

# dict for distribution plots
lb_labels = {2:'Sensorimotor', 4:'Parietal'}

for lb_n, lb in enumerate(zip((1,2,3,1), (2,2,2,4))): # pcs to include, lb code
    
    plt_ind = lb_n
        
    # get some data             
    pre = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[1]) & (pc_df['time'] == 0)][f'PC{lb[0]}']
    pos = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[1]) & (pc_df['time'] == 1)][f'PC{lb[0]}']
    
    pc_min = min(pre.min(), pos.min())
    pc_max = max(pre.max(), pos.max())
     
    bins = np.linspace(np.round(pc_min)-1, np.round(pc_max)+1, 200)
    
    # plot KDE
    pre_kde = gaussian_kde(np.array(pre))
    pos_kde = gaussian_kde(np.array(pos))
    
    kde_bin = np.linspace(np.round(pc_min)-1, np.round(pc_max)+1, 200)
    
    dist_axs[plt_ind].plot(kde_bin, pre_kde(kde_bin), color = colors[0])
    dist_axs[plt_ind].plot(kde_bin, pos_kde(kde_bin), color = colors[1])
    
    # plot hist
    dist_axs[plt_ind].hist(np.array(pre), bins=bins, density=True, alpha=0.7, label='Pre', color = colors[0], zorder=2)
    dist_axs[plt_ind].hist(np.array(pos), bins=bins, density=True, alpha=0.7, label='Post', color = colors[1], zorder=2)
                
    # store extrema for scaling across lobes
    for val in (pre_kde, pos_kde):
        y['min'].append(min(val(kde_bin)))
        y['max'].append(max(val(kde_bin)))
        
    x['min'].append(min(pre.min(), pos.min()))
    x['max'].append(max(pre.max(), pos.max()))

    # plot centile lines and example wave lines
    for cnt_n, cnt in enumerate((10, 50, 90, 97.5, 2.5)):
        ctl = np.percentile(pc_df[(pc_df['cond']==cnd_n) & (pc_df['lobe']==lb[1])][f'PC{lb[0]}'], cnt)
        dist_lim = dist_axs[plt_ind].get_ylim()
        
        if cnt_n < 3:
            dist_axs[plt_ind].axvline(ctl, zorder=1, linestyle='--', color='lightgrey')
        else:
            # add some vlines to ref location of scores for example waves
            dist_axs[plt_ind].axvline(ctl, ymax = 0.5, zorder=1, linestyle='dotted', linewidth=1.5, color='mediumslateblue')            
            
            if lb_n == 0:
                lbl = ['+' if cnt_n == 3 else '\u2013']
                ha = ['left' if cnt_n == 3 else 'right']
                
                y_lim = dist_axs[plt_ind].get_ylim()
                yloc = 0.52*(y_lim[1] - y_lim[0])
                dist_axs[plt_ind].annotate(lbl[0] + r'$^{ve}$ tail', xy=(ctl, yloc), xycoords='data', ha = ha[0])

    # cramer von mises test to compare distributions        
    pre_count = pre.groupby(pd.cut(pre, bins=bins)).size() 
    pos_count = pos.groupby(pd.cut(pos, bins=bins)).size()
    
    res = cramervonmises_2samp(pre_count, pos_count)           
    
    # formatting                                     
    dist_axs[plt_ind].set_xlabel(f'PC{lb[0]} Score', fontsize=12)

    # legend
    if lb_n == 3:
        dist_axs[plt_ind].legend(frameon=False, loc=(-0.4,0.2))
    
    # lobe titles 
    if lb_n == 1 or lb_n == 3:
        dist_axs[plt_ind].set_title(lb_labels[lb[1]], y = 1.03, fontsize=20, color='purple', fontstyle='italic')
    
    # panel label
    if lb_n == 0 or lb_n == 3:
        pn_lbl = [lts[0] if lb_n == 0 else lts[1]]            
        dist_axs[lb_n].annotate(pn_lbl[0], xy=(-0.1, 1.05), 
                              xycoords='axes fraction', 
                              fontsize=25, 
                              fontweight=200)
     
    # annotate CMV test outcome
    if res.pvalue < 0.003125:
        p = 'P < 0.003*'
    else:
        p = 'P = ns'
        
    dist_axs[plt_ind].annotate('\u03c9' + r'$^{2}$' + f'= {np.round(res.statistic, 2)} \n{p}',
                               (0.6, 0.9),
                               xycoords='axes fraction',
                               fontsize=8,
                               color='gray')
    
for ax_n, ax in enumerate(dist_axs):
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_yticks([])
    ax.set_position(ex_wave_loc_dict[ax_n], which='both')
    

    
for lb_n, lb in enumerate(zip(('Sensorimotor', 'Parietal'), (2,4))):
    # add inset for PAIF
    ins_axs = wv_axs[lb_n].inset_axes([0.7,0.7,0.3,0.3])
    ins_axs.spines[['right', 'top']].set_visible(False)
    ins_axs.set_xticks(np.arange(5)*12, fontsize=8, labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])                
    ins_axs.set_ylabel('Inst. Freq (Hz)', fontsize=8)
    ins_axs.tick_params(axis='y', labelsize=8)
    
    # plot example waves
    for tail_n, tail in enumerate(zip((97.5,2.5),('+', '\u2013'), ('--', '-.'))):   # get relevant scores    
        
        sc_pre = []
        sc_pos = []
        
        for pc_n in range(10):
            
            # get PC data             
            pre = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[1]) & (pc_df['time'] == 0)][f'PC{pc_n+1}']
            pos = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[1]) & (pc_df['time'] == 1)][f'PC{pc_n+1}']
            
            # pre wave pcs
            sc_pre.append(np.percentile(pre, 50, axis=0))
            
            # pos wave pcs
            if lb_n == 0 and pc_n < 3:
                sc_pos.append(np.percentile(pos,tail[0], axis=0))
            elif lb_n == 1 and pc_n == 0:
                sc_pos.append(np.percentile(pos,tail[0], axis=0))
            else:
                sc_pos.append(np.percentile(pos,50, axis=0))
        paif = {}
        norm_wvs = {}        
        for sc in (zip((pre_ph_avg, pos_ph_avg), (sc_pre, sc_pos), ('pre', 'pos'))):
            # project score
            proj_dm = pca.project_score(sc[1])
            proj = proj_dm + sc[0].T
            
            paif[sc[2]] = proj_dm
            norm_wvs[sc[2]] = emd.cycles.normalised_waveform(proj[0, :])        
            
        #plot pre
        if tail_n == 0:            
            wv_axs[lb_n].plot(norm_wvs['pre'][1], lw = 2, c = 'darkgrey', ls='dotted')
            wv_axs[lb_n].plot(norm_wvs['pre'][0], lw = 2, c = colors[0], label='Pre')
            
            # ref lines for peak/trough
            for vl in (12, 36):            
                ins_axs.axvline(x=vl, linestyle = ':', color='grey', linewidth=1)
            
            ins_axs.plot(paif['pre'].T, c = colors[0], label = 'Pre')            
            
        # plot pos
        wv_axs[lb_n].plot(norm_wvs['pos'][0], lw = 2, c = colors[1], ls=tail[2], label='Post ('+ tail[1] + r'$^{ve}$ tail)')
        ins_axs.plot(paif['pos'].T, c = colors[1], ls = tail[2], label = 'Post')
        
    # formatting
    wv_axs[lb_n].set_title(lb[0], y = 1.03, fontsize=20, color='purple', fontstyle='italic')
    wv_axs[lb_n].spines[['top', 'right']].set_visible(False)
    wv_axs[lb_n].spines[['bottom', 'left']].set_linewidth(2)
    wv_axs[lb_n].tick_params(width=2)
    wv_axs[lb_n].set_yticks([-1, -0.5, 0, 0.5, 1])
    wv_axs[lb_n].set_xticks(np.linspace(0,48,3),np.linspace(0,1,3))
    wv_axs[lb_n].set_xlabel('Proportion of Cycle', fontsize=15)   

    wv_axs[lb_n].annotate(lts[lb_n+2], xy=(-0.2, 1.05), 
                          xycoords='axes fraction', 
                          fontsize=25, 
                          fontweight=200)             
    
    if lb_n == 0:
        wv_axs[lb_n].set_ylabel('Normalised waveform (au)', fontsize=15)
        wv_axs[lb_n].legend(loc='lower left', frameon=False)

# save
os.chdir(f'{mPath}')
fname = 'Fig 6.jpg'
plt.savefig(fname, dpi=600, bbox_inches='tight') 
        
            
            
            
           
    
