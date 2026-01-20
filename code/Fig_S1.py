# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:04:20 2025

@author: a1147969
"""


import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_uppercase as lts

os.chdir('C:/Users/a1147969/Box/#uni/#Projects/waveform/drugs study/dzp_study/manuscript/submitted/Imag Neurosci/response/Rev2/R2_Q10/')

dat = pickle.load(open('pc_split_out.pkl', 'rb'))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), width_ratios=[1,0.5])

C = dat['cor']
evr = dat['evr']

# variance per split
h1 = axs[0].boxplot(evr[0, :, :].T, positions=2*np.arange(10)-0.3, medianprops={'color': 'green', 'linewidth':3})
h2 = axs[0].boxplot(evr[1, :, :].T, positions=2*np.arange(10)+0.3, medianprops={'color': 'red', 'linewidth':3})
axs[0].set_xlim(-1, 19)
axs[0].set_yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axs[0].set_xticks(np.arange(10)*2, np.arange(1, 11))
axs[0].set_title('Variance Explained')
axs[0].set_ylabel('Proportion variance explained')
axs[0].set_xlabel('Principal Component')
lh = axs[0].legend(labels=['First Half', 'Second Half'], frameon=False)

for n, ind in enumerate(zip((0,1), ('green', 'red'))):
    lh.legend_handles[ind[0]].set_color(ind[1])
    lh.legend_handles[ind[0]].set_linewidth(3)

# correlation between splits
axs[1].boxplot(C[:, :].T, positions=2*np.arange(10))
axs[1].set_xticks(np.arange(10)*2, np.arange(1, 11))
axs[1].set_xlim(-1, 19)
axs[1].set_title('Split-half correlation')
axs[1].set_ylabel('Correlation Coefficient')
axs[1].set_xlabel('Principal Component')

for n, ax in enumerate(axs):
    ax.spines[['right', 'top']].set_visible(False)
    
    
plt.savefig('split_half_validation.jpg', dpi=600, bbox_inches='tight')

