# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 07:40:28 2025

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

from base.params import grp_dat_path, mPath, colors, stat_loc

os.chdir(grp_dat_path)

# create plot, set details
fig = plt.figure(figsize=(8,10), dpi=600)   
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)#, wspace=0.5, hspace=0.2)


subfigs = fig.subfigures(2, 1, height_ratios = [0.4,1],
                         wspace = 0,
                         hspace = 0.3)

# ------------ plot stats -------------- #

# add subplots
stat_axs = subfigs[0].subplots(1, 4, gridspec_kw={'wspace':0.25})

for pc_n, pc in enumerate(range(1, 5, 1)):
    
    # dzp session only
    cnd_n = 1
        
    # load data
    emm = pd.read_csv(f'{mPath}stats/glmm/models/PCs/PC{pc}/PC{pc}_EMMs_3int.csv')
    
    # get cond data
    emm_cnd = emm[(emm['cond']==cnd_n) & (emm['lobe']!=3)]
    
    # scaling for plot
    sc_min = emm_cnd['lower.HPD'].min()
    sc_max = emm_cnd['upper.HPD'].max()
                   
    #get error vals
    emm_cnd['errLw'] = np.abs(emm_cnd['emmean'] - emm_cnd['lower.HPD'])
    emm_cnd['errUp'] = np.abs(emm_cnd['emmean'] - emm_cnd['upper.HPD'])
    
    errLo = emm_cnd.pivot(index='lobe', columns='time', values='errLw')
    errHi = emm_cnd.pivot(index='lobe', columns='time', values='errUp')
    
    err = []
    for co in errLo:
        err.append([errLo[co].values, errHi[co].values])
        
    bp_ax = emm_cnd.pivot(index='lobe', columns='time', values='emmean').plot(kind='bar', 
                                                                              yerr=err, 
                                                                              ax = stat_axs[pc_n],
                                                                              color=colors,
                                                                              width=0.83)
    
    # legend
    if pc_n != 3:
        bp_ax.get_legend().remove()
    else:        
        # format
        handles, labels = bp_ax.get_legend_handles_labels()
        bp_ax.legend(handles, ['Pre', 'Post'], frameon=False, loc='lower left')
        
    # format
    bp_ax.spines[['top', 'right']].set_visible(False)
    bp_ax.spines[['bottom', 'left']].set_linewidth(2)
    bp_ax.tick_params(width=2)
    bp_ax.set_ylabel(f'PC{pc} Score (EMM)', fontsize=12)
    bp_ax.set_xticks((0,1,2,3), ['Temp', 'Occ', 'SM', 'Par'], rotation=45, fontsize=12)
    bp_ax.set_xlabel('')
    
    bp_ax.axhline(linestyle='--', linewidth=1.5, color='gray')
        
    # panel labels                       
    bp_ax.annotate(lts[pc_n], xy=(-0.4, 1.1), xycoords='axes fraction', fontsize=30, fontweight=200) 
        
    # add symbols
    # load ref
    stat = pd.read_csv(f'{mPath}stats/glmm/models/PCs/PC{pc}/stat_map_diazepam.csv', header=None).astype(str)
    
    for st_n, st in enumerate((0,1,2,4)):                
            
        if stat.iloc[0,st_n] != '0':
        
            sgn = np.sign(emm_cnd.iloc[1,4])
        
            temp = emm_cnd[emm_cnd['lobe']==st]                    
            y = [temp.iloc[1,6] if sgn==1 else temp.iloc[1,5]][0]
            
            # scaling
            sc = bp_ax.get_ylim()
            rng = 0.1*np.abs(sc[1] - sc[0])
            
            rng = [y+rng if sgn==1 else y-rng]
            
            bp_ax.annotate('*', xy=(st_n+0.2, rng[0]), fontsize=12, ha='center', va='center_baseline')
            
            # adjust y scale if necessary
            if sgn == 1 and rng > sc[1]:
                bp_ax.set_ylim(bottom=sc[0], top=np.round(rng,1)+0.1)
            elif sgn == -1 and rng < sc[0]:
                bp_ax.set_ylim(bottom=np.round(rng,1)-0.1, top=sc[1])
                    
    fmt = '%.1f'  
    yticks = mtick.FormatStrFormatter(fmt)
    bp_ax.yaxis.set_major_formatter(yticks)
    
    bp_ax.set_position(stat_loc[pc_n], which='both')                
                
#---- plot example norm wave ----#

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

ph_avg = np.array(pf_df.iloc[:,:48].mean(axis=0))

# add axes in subfig
nrm_axs = subfigs[1].subplots(2,2)

for lb_n, lb in enumerate(zip(('temp', 'occ', 'sm', 'par'),(0,1,2,4), ('Temporal', 'Occipital', 'Sensorimotor', 'Parietal'))):
    
    sc_pre = []
    sc_pos = []
    
    for pc_n in range(10):
        
        # get PC data             
        pre = pc_df[(pc_df['cond'] == 1) & (pc_df['time'] == 0) & (pc_df['lobe'] == lb[1])][f'PC{pc_n+1}']
        pos = pc_df[(pc_df['cond'] == 1) & (pc_df['time'] == 1) & (pc_df['lobe'] == lb[1])][f'PC{pc_n+1}']
        
        # pre wave pcs
        sc_pre.append(np.percentile(pre, 50, axis=0))
        
        # pos wave pcs        
        sc_pos.append(np.percentile(pos, 50, axis=0))
        
    paif = {}
    norm_wvs = {}        
    for sc in (zip((sc_pre, sc_pos), ('pre', 'pos'))):
        # project score
        proj_dm = pca.project_score(sc[0])
        proj = proj_dm + ph_avg.T
        
        paif[sc[1]] = proj_dm
        norm_wvs[sc[1]] = emd.cycles.normalised_waveform(proj[0, :])        
    
    # get axis
    axs = nrm_axs.flat[lb_n]
    
    # plot pre           
    axs.plot(norm_wvs['pre'][1], lw = 2, c = 'lightgrey', ls='dotted', label='Sinusoid')
    axs.plot(norm_wvs['pre'][0], lw = 2, c = colors[0], label='Pre')
        
    # plot pos
    axs.plot(norm_wvs['pos'][0], lw = 2, c = colors[1], ls='dashed', label='Post')
    
    # formatting
    axs.set_title(lb[2], loc='right', y = 0.85, fontsize=20, color='purple', fontstyle='italic')
    axs.spines[['top', 'right']].set_visible(False)
    axs.spines[['bottom', 'left']].set_linewidth(2)
    axs.tick_params(width=2)
    axs.set_yticks([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], fontsize=12)
    axs.set_xticks(np.linspace(0,48,3),np.linspace(0,1,3), fontsize=12)
    
    if lb_n > 1:
        axs.set_xlabel('Proportion of Cycle', fontsize=15)   
    
    if lb_n == 0 or lb_n == 2:
        axs.set_ylabel('Normalised waveform (au)', fontsize=15)
    
    if lb_n == 0:
        axs.legend(loc='lower left', fontsize=10, frameon=False)
        
        # panel labels                       
        axs.annotate(lts[4], xy=(-0.22, 1.1), xycoords='axes fraction', fontsize=30, fontweight=200) 
    
# save
os.chdir(f'{mPath}figures/')
fname = 'Fig 5.jpg'    
plt.savefig(fname, dpi=600, bbox_inches='tight') 
