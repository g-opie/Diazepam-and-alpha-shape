"""Determines source locations for SSD components using template matching.
"""
# %% Import packages
import mne
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pickle
import os
from specparam.bands import Bands

from base.params import src_out, ID, ssd_snr_thresh, mPath, mne_datIn, cond

src = False

bands = Bands({'a_frq': [7, 14],
               'lb_frq': [15, 20],
               'ub_frq': [21, 30]})


def find_closest_area(labels, point):
    """Find the closest region from the parcellation for a specified point.

    Parameters
    ----------
    labels : list
        List of labels from mne.read_labels_from_annot.
    point : array
        3D coordinates of a point.

    Returns
    -------
    label : mne.Label
        Closest region.
    """
    trans = mne.read_trans(f'{src_out}/bem/fsaverage-trans.fif')
    point = mne.transforms.apply_trans(trans['trans'], point)
    distances = [np.min(np.sum(np.abs(
        label.pos-point)**2, axis=1)) for label in labels]
    idx = np.argmin(distances)
    return labels[idx]


def compute_pattern_location(df_patterns, sparam_df, fwd, labels, src_loc):
    """Returns a new DataFrame with xyz-location and classification as mu or
    alpha.

    Parameters
    ----------
    df_patterns : pd.DataFrame
        DataFrame as saved by proc_2 ssd procedure.
    fwd : mne.Forward
        Forward model.
    labels : list
        List of labels.

    Returns
    -------
    df_loc : pd.DataFrame
        DataFrame containing location and binary alpha/mu classification.
    """
    # Reorder according to leadfield    
    LF = fwd["sol"]["data"].T
    pos_brain = fwd["source_rr"]    

    # Compute absolute cosine distance
    distances = cdist(LF, df_patterns.T, metric="cosine")
    distances = 1 - np.abs(1 - distances)

    # Return xyz-coordinates of minimum distance node
    idx = np.argmin(distances, axis=0)
    mni_pos = pos_brain[idx]

    regions = []
    for point in mni_pos:
        region = find_closest_area(labels, point)
        regions.append(region.name)

    # Create a new DataFrame, classify lobe of each source
    new_col = {'x':mni_pos[:, 0], 'y':mni_pos[:, 1], 'z':mni_pos[:, 2], 'region':regions, 'LF_node':idx, 'dist':np.min(distances, axis=0)}
    df_loc = sparam_df.assign(**new_col)
    df_loc.insert(df_loc.shape[1],'lobe', "")
    
    # check if location is included in target lobes, specified in 'src_loc' variable    
    for reg in src_loc:
        src_regs = '|'.join(src_loc[reg])
        src_bool = df_loc.region.str.contains(src_regs)
        
        for x in range(len(src_bool)):
            if src_bool[x] == True:
                df_loc.iloc[x, df_loc.shape[1]-1] = reg   
    
    return df_loc

def get_sources_one_subj(participant, fwd, labels, src_loc):
    """Determines source locations for SSD components using template matching
    for one subject.

    Parameters
    ----------
    participant : str
        Subject ID.
    fwd : mne.Forward
        Forward model.
    labels : list
        List of labels.

    Returns
    -------
    source_df : pd.DataFrame
        DataFrame with source locations.
    """
    
    os.chdir(f'{mne_datIn}{participant}/')
    
    # get subj ssd 
    ssd_out = pickle.load(open(f'ssd/{participant}_ssd_out.pkl', 'rb'))
    
    for cnd_id, cnd in enumerate(cond):
        for bnd_id, (bnd_nm, bnd_rng) in enumerate(bands):            
            
            # get pattern
            patt = ssd_out[f'{cnd}_{bnd_nm}']['pattern']
            
            if np.isnan(patt).any():
                continue
            else:
                # get ssd spect data
                sparam_df = pd.read_csv(f'ssd/{cnd}_ssd_spect_{bnd_nm}.csv')
                
                # set pattern as df
                patt = pd.DataFrame(patt)
                
                # get pattern location
                source_df = compute_pattern_location(patt, sparam_df, fwd, labels, src_loc)

                # Filter components based on specparam pk, src dist, SNR
                source_df = source_df[(source_df['Freq.'].notnull()) &
                                      (source_df['dist'] < 0.15) &
                                      (source_df['SNR'] > ssd_snr_thresh[bnd_nm])]
                
                # remove srcs localisaed outside of target cortices
                source_df = source_df[source_df['lobe'].astype(bool)]
                
                if not os.path.isdir(f'{mne_datIn}/{participant}/sources/'):
                    os.makedirs(f'{mne_datIn}/{participant}/sources/')
                
                # save
                source_df.to_csv(f'sources/{participant}_{cnd}_{bnd_nm}_srcs.csv', index=False)

def get_sources_all_subjs(save_fname=src_out, 
                          parcellation_subj='fsaverage',
                          annotation='HCPMMP1'):
    """Determines source locations for SSD components using template matching
    for all subjects.

    Parameters
    ----------
    save_fname : str
        Path to save CSV file with source locations.
    mri_folder : str
        Path to MRI folder.
    parcellation_subj : str
        Name of subject with parcellation.
    annotation : str
        Name of annotation.

    Returns
    -------
    df_all : pd.DataFrame
        DataFrame with source locations.
    """    

    # load template leadfield
    fwd_fname = src_out + 'fwd_sol-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)    

    # src locs
    src_loc = pickle.load(open(f'{mPath}source/src_locs.pkl', 'rb'))
    
    # load labels
    labels = mne.read_labels_from_annot(parcellation_subj, annotation)
    
    del labels[:2]

    # Load data
    for i_sub, participant in enumerate(ID):
        print(f'Subject {participant} ({i_sub + 1}/{len(ID)})')
        
        get_sources_one_subj(participant, fwd, labels, src_loc)


if __name__ == '__main__':
    
    get_sources_all_subjs()