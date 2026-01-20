# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 12:42:08 2025

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

from params import mne_datIn, ID, time, grp_dat_path, cond, mPath, colors, numPCs, cnd_lab

fig = plt.figure(figsize=(4,8), dpi=600)   
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.2)

subfig = fig.subfigures(1,3, width_ratios=[1,0.5,1])

for cnd_n, cnd in enumerate(cond):
    
    if cnd_n == 0:
        fig_ind = 0
    else:
        fig_ind = 2
        
    axs = subfig[fig_ind].subplots(4,1)
    
    # add cond label
    axs[0].set_title(cnd_lab[cnd_n], y = 1.1, fontsize=20, fontweight=525, color='purple', fontstyle='italic')
    
    for pc_n in range(numPCs):        
        
        # load data
        emm = pd.read_csv(f'{mPath}stats/glmm/models/PCs/PC{pc_n+1}/PC{pc_n+1}_EMMs_3int.csv')
        
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
                                                                                  ax = axs[pc_n],
                                                                                  color=colors,
                                                                                  width=0.83)
        
        bp_ax.axhline(linestyle='--', linewidth=1.5, color='gray')  
        
        # legend
        if pc_n == 0 and cnd_n == 0:
            # format
            handles, labels = bp_ax.get_legend_handles_labels()
            bp_ax.legend(handles, ['Pre', 'Post'], frameon=False)
        else:        
            bp_ax.get_legend().remove()
            
        # format
        bp_ax.spines[['top', 'right']].set_visible(False)
        
        if cnd_n == 0:
            bp_ax.set_ylabel(f'PC{pc_n+1} Score (EMM)')
        
        if pc_n < numPCs-1:
            bp_ax.set_xlabel('')
            bp_ax.set_xticks((0,1,2,3),['','','',''])
        else:
            bp_ax.set_xlabel('Lobe')            
            bp_ax.set_xticks((0,1,2,3), ['Temp', 'Occ', 'SM', 'Par'], rotation=45)
            
        # panel label
        ls = lts[pc_n + (cnd_n*4)]                    
        bp_ax.annotate(ls, xy=(-0.35, 1.05), xycoords='axes fraction', fontsize=22, fontweight=200) 
            
        # add symbols
        if cnd_n == 1:
            
            # load ref
            stat = pd.read_csv(f'{mPath}stats/glmm/models/PCs/PC{pc_n+1}/stat_map_{cond[cnd_n]}.csv', header=None).astype(str)
            
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
                    elif sgn == -1 and rng < sc[1]:
                        bp_ax.set_ylim(bottom=np.round(rng,1)-0.1, top=sc[1])
                        
        fmt = '%.1f'  
        yticks = mtick.FormatStrFormatter(fmt)
        bp_ax.yaxis.set_major_formatter(yticks)
        
# save fig
fig.savefig(f'{mPath}figures/stats_by_area_supp_fig.png', dpi=600, bbox_inches='tight')