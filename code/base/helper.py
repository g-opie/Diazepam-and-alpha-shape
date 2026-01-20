"""Relatively general-purpose functions used in multiple scripts.
"""
# Import packages
import os.path
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mat
import seaborn as sns
import mne
import specparam as spec
from specparam import SpectralGroupModel
import ssd
from params import src_out, mne_datIn
import scipy.linalg
from scipy.signal import spectrogram as spec_gr
import numpy as np
import more_itertools as mit
# import fooof
import pickle


def add_sym_btwn (axs, stat_map, clr='k', lnwd=2):
    
    # get data from plot
    chd = axs.get_children()    
    lines = [chd[x] for x in range(len(chd)) if isinstance(chd[x], mat.collections.LineCollection)]
    
    ydat = [chd[x] for x in range(len(chd)) if isinstance(chd[x], mat.lines.Line2D)]
    
    # contrast idx
    idx = np.arange(len(lines))
    idx = np.reshape(idx, (len(stat_map[0], 2)))    
    
    for pts in range(idx.shape[0]):        
               
        # line data
        err1 = lines[idx[pts,0]].get_segments()
        err2 = lines[idx[pts,1]].get_segments()
        
        err = np.vstack((err1, err2))
        
        # largest yerr-value
        yerr_max = np.max(err[:,1])
        
        # x values for data point
        x1 = np.min(err[:,0])
        x2 = np.max(err[:,0])
        
        # y values for data point
        y1 = ydat[idx[pts,0]].get_ydata()
        y2 = ydat[idx[pts,1]].get_ydata()
        
        # between group marker 
        if np.isnan(stat_map[0][pts]):
            continue
        else:            
            axs.hlines(1.1*yerr_max, x1, x2, color=clr, linewidth=lnwd)
            axs.annotate('*', xy=(x2-x1, 1.15*yerr_max), fontsize=12, fontweight='semibold')
    
        # within group marker [grp1]
        if np.isnan(stat_map[1][pts]):
            continue
        else:            
            axs.plot(0.8*x1, 1.15*y1, marker=stat_map[1][pts], c='k', mew=0.5, ms=8)

        # within group marker [grp2]
        if np.isnan(stat_map[2][pts]):
            continue
        else:            
            axs.plot(1.2*x2, 1.15*y2, marker=stat_map[2][pts], c='k', mew=0.5, ms=8)
            
    return axs


def get_dummy_codes(dat, sub=0, time=0, cnd=0, lbe=0, ssd='0'):
    
    n = dat.shape[0]
    
    if cnd == 'placebo':
        cnd = 0
    elif cnd == 'diazepam':
        cnd = 1
        
    coded_dat = np.hstack((dat,
                           np.full((n,1), int(sub)),
                           np.full((n,1), time),
                           np.full((n,1), cnd),
                           np.full((n,1), lbe),
                           np.reshape(np.arange(0,n,1),(n,1)),
                           np.full((n,1), ssd)))
    
    return coded_dat

def ind_sub_exponent(subj, cond, time, ep_length = 4):
    raw = load_raw_eeg(subj, cond, time, IC_dat=True, avg_ref=True)
    
    raw_events = mne.make_fixed_length_events(raw, duration=ep_length)
    
    ep = mne.Epochs(raw, raw_events, tmin=0, tmax=ep_length, baseline=None)
    
    ep_psd = ep.compute_psd(method='welch', fmin = 2, fmax = 40, n_fft=2000, n_overlap=1000, tmin=0, tmax=ep_length).get_data(return_freqs=True)
    
    spect_out = []
    
    #run specparam over epochs
    for ep_id in range(len(ep_psd[0])):

        #Initialize SpectralGroupModel object
        fg = SpectralGroupModel(peak_width_limits=[1, 12],
                                max_n_peaks=6, 
                                min_peak_height=0.2, 
                                verbose=False)
    
        #Define frequency range, fit model, plot results
        freq_range = [2, 40]
        fg.fit(ep_psd[1], ep_psd[0][ep_id], freq_range)
        
        spect_out.append(fg.get_params('aperiodic_params', col='exponent'))
        
        
    return spect_out

def find_osc(subj, cond, time, comp):
    
    # dat, _ = load_comp_timeseries(subj, cond, time)
    
    # dat = dat[comp]
    
    #get spectrogram
    f, t, sx = spec_gr(comp, fs=1000, nperseg=500, noverlap=250)
    
    #get mean for alpha, find where signal exceeds mean
    a_avg = np.mean(sx[4:6,:])
    
    filt = np.where(np.mean(sx[4:6,:], axis=0) > a_avg)[0]
    
    #find longest section above average
    long_chain = max((list(g) for g in mit.consecutive_groups(filt)), key=len)
    
    idx_smp = ((long_chain[-1] - long_chain[0])/2) + long_chain[0]
    
    idx_tme = t[int(idx_smp)]*1000
    
    return int(idx_tme)


def get_sign(p):    
    idx = np.argmax(np.abs(p))
    sign = np.sign(p[idx])
    
    return sign

def reject_bad_segs(raw):
    """ This function rejects all time spans annotated as annot_to_reject and concatenates the rest"""
    
    if len(raw.annotations) == 0: #if no annotated sections
        return raw    
    else:
        raw_segs = []
        for jsegment in range(1, len(raw.annotations)+1):
            if jsegment == len(raw.annotations):
                if raw.annotations.onset[jsegment-1] + raw.annotations.duration[jsegment-1] == raw.times[-1]:
                        continue
                else:
                    raw_segs.append(raw.copy().crop(tmin = raw.annotations.onset[jsegment-1]+raw.annotations.duration[jsegment-1], 
                                                    tmax = raw.times[-1], 
                                                    include_tmax = True))
            
            elif jsegment < len(raw.annotations) and raw.annotations[jsegment]['description'] == 'bad':
                
                if jsegment == 1:
                    if raw.annotations.onset[jsegment-1] == 0:
                        continue
                    else:
                        raw_segs.append(raw.copy().crop(tmin=0, tmax=raw.annotations.onset[jsegment-1], include_tmax=False))
                
                tmin = raw.annotations.onset[jsegment-1] + raw.annotations.duration[jsegment-1] # start at ending of last bad annot
                tmax = raw.annotations.onset[jsegment] # end at onset of current bad annot
                raw_segs.append(
                    raw.copy().crop(tmin=tmin,
                                    tmax=tmax,
                                    include_tmax=False))
            
    
    
        return mne.concatenate_raws(raw_segs)

def get_source_labels(
        annotation='HCPMMP1', subject='fsaverage', names=True):
    """Label alpha and mu sources.

    Parameters
    ----------
    annotation : str
        Name of annotation file for desired parcellation.
    subject : str
        Name of subject for desired parcellation.
    mri_folder : str
        Path to folder containing MRI data.
    names : bool
        If True, returns names of labels instead of label objects.

    Returns
    -------
    alpha_selected : list
        List of labels for alpha sources.
    mu_selected : list
        List of labels for mu sources.
    """
    labels = mne.read_labels_from_annot(
        subject, annotation)

    # file is from:
    # https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/atlases.html
    # contains data frame with classifications for each region
    df = pd.read_csv(src_out + 'regions.csv')

    # select sensory-motor labels
    labels_roi = df.query('cortex == "Somatosensory_and_Motor" or cortex == '
                          '"Paracentral_Lobular_and_Mid_Cingulate" or cortex '
                          '== "Premotor"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]
    sm_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name.startswith(labels_roi1 + '_'):
                if names:
                    sm_selected.append(label.name)
                    continue
                sm_selected.append(label)

    assert len(labels_roi) == len(sm_selected)    

    # select occipital labels
    labels_roi = df.query('Lobe=="Occ"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]    
    occ_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name in sm_selected:
                continue
            else:
                if label.name.startswith(labels_roi1 + '_'):
                    if names:
                        occ_selected.append(label.name)
                        continue
                    occ_selected.append(label)

    assert len(labels_roi) == len(occ_selected)
    
    # select parietal labels
    labels_roi = df.query('Lobe == "Par"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]    
    par_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name in sm_selected:
                continue
            else:
                if label.name.startswith(labels_roi1 + '_'):
                    if names:
                        par_selected.append(label.name)
                        continue
                    par_selected.append(label)

    # assert len(labels_roi) == len(par_selected)
    
    # select temporal labels
    labels_roi = df.query('Lobe == "Temp"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]
    temp_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name in sm_selected:
                continue
            else:
                if label.name.startswith(labels_roi1 + '_'):
                    if names:
                        temp_selected.append(label.name)
                        continue
                    temp_selected.append(label)

    assert len(labels_roi) == len(temp_selected)
    
    # select frontal labels
    labels_roi = df.query('Lobe == "Fr"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]
    fr_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name in sm_selected:
                continue
            else:
                if label.name.startswith(labels_roi1 + '_'):
                    if names:
                        fr_selected.append(label.name)
                        continue
                    fr_selected.append(label)

    # assert len(labels_roi) == len(fr_selected)

    return sm_selected, occ_selected, par_selected, temp_selected, fr_selected


def load_ssd(participant_id, cond, results_folder=mne_datIn):
    """Load spatial filters and patterns for a specific dataset.

    Parameters
    ----------
    participant : str
        Participant ID.
    results_folder : str
        Path to folder containing results.

    Returns
    -------
    patterns : np.ndarray, 2D
        Spatial patterns as computed by SSD.
    filters : np.ndarray, 2D
        Spatial filters as computed by SSD.
    """
    # Load filters
    ssd_filters_fname = f"{results_folder}/{participant_id}/{participant_id}_{cond}_filt.csv"
    filters_df = pd.read_csv(ssd_filters_fname)
    filters = filters_df.values.T

    # Load patterns
    ssd_patterns_fname = ssd_filters_fname.replace('filt', 'patterns')
    patterns_df = pd.read_csv(ssd_patterns_fname)
    patterns = patterns_df.values.T
    return patterns, filters

def apply_laplacian(raw, channels):
    """Use a Laplacian over desired channels using each channel's neighbors.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance to apply Laplacian to.
    channels : dict
        Keys are picked channels, while values are lists of neighboring
        channels.

    Returns
    -------
    raw : instance of Raw
        Raw instance with Laplacian applied to picked channels.

    """
    # Make copy of original to prevent mix-ups
    orig_raw = raw.copy()

    # Subtract average of neighbors from each picked channel
    ch_lst, processed_data = [], []
    for ch, neighbors in channels.items():
        ch_data = orig_raw.get_data(picks=ch)
        neighbor_data = orig_raw.get_data(picks=neighbors)

        processed_data.append(ch_data - np.mean(neighbor_data, axis=0))
        ch_lst.append(ch)

    # Concatenate all processed channel data
    processed_arr = np.concatenate(processed_data)

    # Make Raw instance from data
    info = orig_raw.pick_channels(ch_lst).info
    processed_raw = mne.io.RawArray(processed_arr, info)
    return processed_raw


def get_SNR(raw, fmin=2, fmax=40, seconds=10, freq=[8, 13]):
    """Compute power spectrum and calculate 1/f-corrected SNR in one band.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing traces for which to compute SNR
    fmin : float
        minimum frequency that is used for fitting spectral model.
    fmax : float
        maximum frequency that is used for fitting spectral model.
    seconds: float
        Window length in seconds, converts to FFT points for PSD calculation.
    freq : list
        SNR in that frequency window is computed.

    Returns
    -------
    SNR : array, 1D
        Contains SNR (1/f-corrected, for a chosen frequency) for each channel.
    """
    # Compute PSD
    SNR = np.zeros((len(raw.ch_names),))
    
    raw_events = mne.make_fixed_length_events(raw, duration=10)    
    ep = mne.Epochs(raw, raw_events, tmin=0, tmax=10, baseline=None)    
    ep_psd = ep.compute_psd(method='multitaper', fmin = 2, fmax = 40, tmin=0, tmax=10).get_data(return_freqs=True)
    
    fre = ep_psd[1]
    pwr = np.mean(ep_psd[0], axis=0)

    # Fit aperiodic activity (using same params as control analyses)
    fg = SpectralGroupModel(peak_width_limits=[1, 12],
                            max_n_peaks=6, 
                            min_peak_height=0.1,
                            peak_threshold=1.2,
                            verbose=False)    
    
    fg.fit(fre, pwr, freq_range = [2, 40])

    # Compute aperiodic-corrected SNR
    for pick in range(len(raw.ch_names)):
        psd_corr = 10 * np.log10(pwr[pick]) - 10 * fg.get_model(pick)._ap_fit
        idx = np.where((fre > freq[0]) & (fre < freq[1]))[0]
        idx_max = np.argmax(psd_corr[idx])
        SNR[pick] = psd_corr[idx[0]+idx_max]
        
        
    return SNR, fg


def load_raw_eeg(subj, cond, time, mne_raw_folder=mne_datIn, IC_dat = False, avg_ref = False):
    """Load raw EEG data for a specific subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    mne_raw_folder : str
        Path to folder containing raw MNE files.

    Returns
    -------
    raw : instance of Raw
        Raw instance containing EEG data for a specific subject.
    """
    # Determine file name of raw EEG
    if IC_dat:
        raw_fname = f'{mne_raw_folder}/{subj}/ICA/{subj}_{cond}_{time}_filtCh_IC_raw.fif'
        rej_annot = 'omit'
    else:
        raw_fname = f'{mne_raw_folder}/{subj}/pre_proc/{subj}_{cond}{time}_filtCh_raw.fif'
        rej_annot = None

    # Load raw
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    if avg_ref:
        raw.set_eeg_reference(ref_channels='average')
    
    # # Reject annotted sections, if IC data used
    # raw = raw.get_data(reject_by_annotation=rej_annot)   

        
    return raw

def load_elec_timeseries(subj, elecs, cond, time, IC_dat = False, app_laplacian = False, lap_dict = None):
    """
    return timeseries for specific electrode for individual participants, conditions and time points
    
    Parameters
    ----------
    subj : int
        ID number ofr subject to load
    elecs : str
        which electrodes to load
    cond : str
        condition descriptor
    time : str
        time descriptor
    IC_dat : Bool, optional
        Use data that has undergone ICA cleaning. The default is False.
    app_laplacian : Bool, optional
        Retrun laplacian for target channel. Requires target and neighbouring channels to be defined in 'lap_dict'. The default is False.
    lap_dict : dict, optional
        defintion of channels for laplacian. format as dict with key as target channel and entry as list of neighbouring channels The default is None.

    Returns
    -------
    raw_out : TYPE
        DESCRIPTION.

    """
    # load eeg
    raw = load_raw_eeg(subj, cond, time, IC_dat)
    
    # get timeseries from specified elecs
    if app_laplacian:        
        raw_out = apply_laplacian(raw, lap_dict)     
    
    else: 
        raw_out = raw.get_data(picks = elecs)
        
    return raw_out


def load_comp_timeseries(subj, cond, time, mne_raw_folder=mne_datIn):
        # save_folder=COMP_TIMESERIES_FOLDER, num_components=NUM_COMPONENTS,
        # f_bandpass=F_BANDPASS, s_freq=S_FREQ):
    """Load component time series for a specific subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    mne_raw_folder : str
        Path to folder containing raw MNE files.
    save_folder : str
        Path to folder where component time series should be saved.
    num_components : int
        Number of components to load.
    f_bandpass : tuple
        Bandpass filter to apply to components (fmin, fmax).
    s_freq : float
        Sampling frequency of raw EEG data.

    Returns
    -------
    comps : np.ndarray, 2D
        Component time series with shape (n_components, n_timepoints).
    s_freq : float
        Sampling frequency of component time series.
    """

    # Load component time series from save if already saved seperately
    save_fname = f'{mne_raw_folder}/{subj}/{subj}_{cond}_ssd_comps.npy'
    # if os.path.exists(save_fname):
    #     return np.load(save_fname)

    # Determine filenames for SSD, raw, and figure to be saved
    raw_fn = f'{mne_raw_folder}/{subj}/{subj}_{cond}{time}_filtCh_raw.fif'

    # Load raw file
    raw = mne.io.read_raw_fif(raw_fn)
    raw.load_data()    
    raw.pick_types(eeg=True)

    # Load SSD file
    _, filters = load_ssd(subj, cond)

    # Apply filters to get components
    comps_raw = ssd.apply_filters(raw, filters)
    comps = comps_raw.get_data()

    # Save components to speed up processing next time
    # np.save(save_fname, comps)
    
    return comps, comps_raw


def get_alpha_mu_comps(sources_csv=src_out):
    """Extract alpha and mu components from sources CSV.

    Parameters
    ----------
    sources_csv : str
        Path to CSV containing source localization results.

    Returns
    -------
    alpha_comps : pd.DataFrame
        DataFrame containing alpha components.
    mu_comps : pd.DataFrame
        DataFrame containing mu components.
    """
    # Read in CSV of pattern distances
    df = pd.read_csv(sources_csv)

    # Seperate out mu and alpha components
    alpha_comps = df.query("alpha")[['subject', 'idx', 'SNR', 'pattern_dist']]
    mu_comps = df.query("mu")[['subject', 'idx', 'SNR', 'pattern_dist']]
    return alpha_comps, mu_comps