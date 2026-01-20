# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:56:42 2025

@author: a1147969
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from string import ascii_uppercase as lts
from matplotlib import ticker as mtick
import pandas as pd
from scipy.stats import gaussian_kde, cramervonmises_2samp

from params import  mPath, colors, numPCs, cond   

# load pc data
os.chdir('C:/Users/a1147969/Box/#uni/#Projects/waveform/drugs study/dzp_study/group dat/emd/')
pc_df = pd.read_csv('pc_scores_df.csv')
pc_df = pc_df[pc_df['good_trials'] == True]

exp_stats = 1

stats_out = []

for cnd_n, cnd in enumerate(cond):
    
    # create plot, set details
    fig = plt.figure(figsize=(7,10), dpi=600)   
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.3)

    subfigs = fig.subfigures(4, 1, 
                             wspace = 0.05,
                             hspace = 0.3)    
    
    for pc_n in range(numPCs):                       
        
        #---- plot PC score distribution ----#
        pc_axs = subfigs[pc_n].subplots(1,4)
        
        y = {'min':[], 'max':[]}
        x = {'min':[], 'max':[]}
        
        for lb_n, lb in enumerate(zip((0,1,2,4), ('Temporal', 'Occipital', 'Sensorimotor', 'Parietal'))):

            # get some data             
            pre = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[0]) & (pc_df['time'] == 0)][f'PC{pc_n+1}']
            pos = pc_df[(pc_df['cond'] == cnd_n) & (pc_df['lobe']==lb[0]) & (pc_df['time'] == 1)][f'PC{pc_n+1}']
            
            pc_min = min(pre.min(), pos.min())
            pc_max = max(pre.max(), pos.max())
             
            bins = np.linspace(np.round(pc_min)-1, np.round(pc_max)+1, 200)
            
            # plot KDE
            pre_kde = gaussian_kde(np.array(pre))
            pos_kde = gaussian_kde(np.array(pos))
            
            kde_bin = np.linspace(np.round(pc_min)-1, np.round(pc_max)+1, 200)
            
            pc_axs[lb_n].plot(kde_bin, pre_kde(kde_bin), color = colors[0])
            pc_axs[lb_n].plot(kde_bin, pos_kde(kde_bin), color = colors[1])
            
            # plot hist
            pc_axs[lb_n].hist(np.array(pre), bins=bins, density=True, alpha=0.7, label='Pre', color = colors[0], zorder=2)
            pc_axs[lb_n].hist(np.array(pos), bins=bins, density=True, alpha=0.7, label='Post', color = colors[1], zorder=2)
                        
            # store extrema for scaling across lobes
            for val in (pre_kde, pos_kde):
                y['min'].append(min(val(kde_bin)))
                y['max'].append(max(val(kde_bin)))
                
            x['min'].append(min(pre.min(), pos.min()))
            x['max'].append(max(pre.max(), pos.max()))
            
            if pc_n == 0:
                pc_axs[lb_n].set_title(lb[1], color='purple', fontstyle='italic')            
            elif pc_n == numPCs-1:
                pc_axs[lb_n].set_xlabel(lb[1])
    
            # plot centile lines
            for cnt_n, cnt in enumerate((10, 50, 90)):
                ctl = np.percentile(pc_df[(pc_df['cond']==cnd_n) & (pc_df['lobe']==lb[0])][f'PC{pc_n+1}'], cnt)
                dist_lim = pc_axs[lb_n].get_ylim()
                
                pc_axs[lb_n].axvline(ctl, zorder=1, linestyle='--', color='lightgrey')    
    
            # cramer von mises test to compare distributions        
            pre_count = pre.groupby(pd.cut(pre, bins=bins)).size() 
            pos_count = pos.groupby(pd.cut(pos, bins=bins)).size()
            
            res = cramervonmises_2samp(pre_count, pos_count)           
            
            # formatting                         
            if pc_n == numPCs-1:
                pc_axs[lb_n].set_xlabel('PC Score')
             
            if pc_n == 0 and lb_n == 3:
                pc_axs[lb_n].legend(frameon=False, loc=(-0.32,0.2))
                
            if lb_n == 0:
                # annotate PC
                pc_axs[lb_n].annotate(f'PC{pc_n+1}', xy=(-0.1, 0.5), 
                                     xycoords='axes fraction', 
                                     fontsize=20, 
                                     fontweight=535,
                                     color='mediumslateblue',
                                     rotation=90,
                                     va='center')
                
                # annotate panel  label
                pc_axs[lb_n].annotate(lts[pc_n], xy=(-0.1, 1.05), 
                                      xycoords='axes fraction', 
                                      fontsize=25, 
                                      fontweight=200)
             
            # annotate CMV test outcome
            if res.pvalue < 0.003125:
                p = 'P < 0.003*'
            else:
                p = 'P = ns'
                
            pc_axs[lb_n].annotate('\u03c9' + r'$^{2}$' + f'= {np.round(res.statistic, 2)} \n{p}',
                                  (0.6, 0.9),
                                  xycoords='axes fraction',
                                  fontsize=8,
                                  color='gray')
            
            stats_out.append([res.pvalue, res.statistic, pc_n, cnd, lb[1]])
            
        for ax in pc_axs.flat:
            ax.spines[['left', 'top', 'right']].set_visible(False)
            ax.set_yticks([])
            ax.set_ylim(min(y['min']), max(y['max']))
            ax.set_xlim(min(x['min']), max(x['max']))
            
    
            
    # save
    os.chdir(f'{mPath}figures/')
    fname = f'stats_by_lobe_{cnd}_supp.jpg'    
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    
if exp_stats == 1:
    df = pd.DataFrame(np.stack(stats_out, axis=0), columns=['pval', 'stat', 'PC', 'cond', 'lobe'])
    df.to_csv(f'{mPath}stats/spect/distribution_stats/by_area_stats.csv', index=False)
    

            