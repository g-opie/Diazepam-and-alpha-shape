# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:34:06 2025

@author: a1147969

get component SNR

"""

import os
import numpy as np
import pandas as pd
from specparam.bands import Bands

from base.params import mne_datIn, cond, ID

#Define bands for SSD
bands = Bands({'a_frq': [7, 14],
               'lb_frq': [15, 20],
               'ub_frq': [21, 30]})

dat = {'a_frq': [],
       'lb_frq': [],
       'ub_frq': []}

for sub in range(len(ID)):
    
    subOut = mne_datIn + ID[sub] + '/ssd/'    
    os.chdir(subOut)    
    
    for cnd_id, cnd in enumerate(cond):
        for bnd_n, (bnd_nm, bnd_rng) in enumerate(bands):
            
            try:
                pks = pd.read_csv(f'{cnd}_ssd_spect_{bnd_nm}.csv')
                
                pks = pks.dropna()
                
                dat[bnd_nm].append(np.array(pks['SNR']))
                
            except FileNotFoundError:
                print('{} not found, moving to next band'.format(bnd_nm))
            
