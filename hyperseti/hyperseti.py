import cupy as cp
import numpy as np
import time
import pandas as pd
import logging
import os

from astropy import units as u
import setigen as stg

from cupyx.scipy.ndimage import uniform_filter1d

from .peak import prominent_peaks, prominent_peaks_optimized

from .data import from_fil, from_h5

import hdf5plugin
import h5py
from copy import deepcopy

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import dask
import dask.bag as db
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

#logging
from .log import logger_group, Logger
from .plotting import imshow_dedopp, imshow_waterfall, overlay_hits

logger = Logger('hyperseti.hyperseti')
logger_group.add_logger(logger)


# Max threads setup
os.environ['NUMEXPR_MAX_THREADS'] = '8'


dedoppler_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKernel
        (const float *data, float *dedopp, int *shift, int F, int T)
        /* Each thread computes a different dedoppler sum for a given channel

         F: N_frequency channels
         T: N_time steps

         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T
        */
        {

        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials

        // Index for output array
        const int dd_idx = d * F + tid;
        float dd_val = 0;

        int idx = 0;
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                dd_val += data[idx];
              }
              dedopp[dd_idx] = dd_val;
            }
        }
''', 'dedopplerKernel')

dedoppler_kurtosis_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKurtosisKernel
        (const float *data, float *dedopp, int *shift, int F, int T, int N)
        /* Each thread computes a different dedoppler sum for a given channel

         F: N_frequency channels
         T: N_time steps
         N: N_acc number of accumulations averaged within time step

         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T

        Note: output needs to be scaled by N_acc, number of time accumulations
        */
        {

        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials

        // Index for output array
        const int dd_idx = d * F + tid;
        float S1 = 0;
        float S2 = 0;

        int idx = 0;
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                S1 += data[idx];
                S2 += data[idx] * data[idx];
              }
              dedopp[dd_idx] = (N*T+1)/(T-1) * (T*(S2 / (S1*S1)) - 1);
            }
        }
''', 'dedopplerKurtosisKernel')

suppression_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void suppressionKernel
        (const float* snrs, const int* drifts, const int* channels, const int* sizes, const int* start, const int* end, bool* save, float* debug)
        /* 
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        
        const int start_idx = start[tid];
        const int end_idx = end[tid];
        
        
        for(int idx = start_idx; idx < end_idx; idx++){
            if(save[idx]){
                int boxcar_size = sizes[idx];
                int end_channel = channels[idx] + boxcar_size + 1;
                int start_channel = channels[idx] - boxcar_size - 1;
                int min_drift = drifts[idx] - boxcar_size - 1;
                int max_drift = drifts[idx] + boxcar_size + 1;
                float snr = snrs[idx];
                debug[idx] = snrs[idx];
                
                int curr_idx = idx+1;
                while(curr_idx < end_idx && channels[curr_idx] <= end_channel){
                    if((drifts[curr_idx] >= min_drift) && (drifts[curr_idx] <= max_drift) && (snrs[curr_idx] < snr)){
                        save[curr_idx] = false;
                    }
                    curr_idx+=1;
                }
                curr_idx = idx-1;
                while(curr_idx >= start_idx && channels[curr_idx] >= start_channel){
                    if((drifts[curr_idx] >= min_drift) && (drifts[curr_idx] <= max_drift) && (snrs[curr_idx] < snr)){
                        save[curr_idx] = false;
                    }
                    curr_idx-=1;
                }
            }
        }
        }
        
''', 'suppressionKernel')


def normalize(data, mask=None, padding=0, return_space='cpu'):
    """ Apply normalization on GPU

    Applies normalisation (data - mean) / stdev

    Args:
        data (np/cp.array): Data to preprocess
        mask (np.cp.array): 1D Channel mask for RFI flagging
        padding (int): size of edge region to discard (e.g. coarse channel edges)
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space

    Returns: d_gpu (cp.array): Normalized data
    """

    # Copy over to GPU if required
    d_gpu = cp.asarray(data.astype('float32', copy=False))
    d_gpu_flagged = cp.asarray(data.astype('float32', copy=True))

    paddingu = None if padding == 0 else -padding

    # Need to correct stats
    N_flagged = 0
    N_tot     = np.product(d_gpu[..., padding:paddingu].shape)
    if mask is not None:
        # Convert 1D-mask to match data dimensions
        mask_gpu = cp.repeat(cp.asarray(mask.reshape((1, 1, len(mask)))), d_gpu.shape[0], axis=0)
        cp.putmask(d_gpu_flagged, mask_gpu, 0)
        N_flagged = mask_gpu[..., padding:paddingu].sum()

    # Normalise
    t0 = time.time()
    # Compute stats based off flagged arrays
    d_mean = cp.mean(d_gpu_flagged[..., padding:paddingu])
    d_std  = cp.std(d_gpu_flagged[..., padding:paddingu])
    flag_correction =  N_tot / (N_tot - N_flagged)
    d_mean = d_mean * flag_correction
    d_std  = d_std * np.sqrt(flag_correction)
    logger.debug(f"flag fraction correction factor: {flag_correction}")

    #  Apply to original data
    d_gpu = (d_gpu - d_mean) / d_std
    t1 = time.time()
    logger.info(f"Normalisation time: {(t1-t0)*1e3:2.2f}ms")

    if return_space == 'cpu':
        return cp.asnumpy(d_gpu)
    else:
        return d_gpu


def apply_boxcar(data, boxcar_size, axis=1, mode='mean', return_space='cpu'):
    """ Apply moving boxcar filter and renormalise by sqrt(boxcar_size)

    Boxcar applies a moving MEAN to the data.
    Optionally apply sqrt(N) factor to keep stdev of gaussian noise constant.

    Args:
        data (np/cp.array): Data to apply boxcar to
        boxcar_size (int): Size of boxcar filter
        mode (str): Choose one of 'mean', 'mode', 'gaussian'
                    Where gaussian multiplies by sqrt(N) to maintain
                    stdev of Gaussian noise
        return_space ('cpu' or 'gpu'): Return in CPU or GPU space

    Returns:
        data (np/cp.array): Data after boxcar filtering.
    """
    if mode not in ('sum', 'mean', 'gaussian'):
        raise RuntimeError("Unknown mode. Only modes sum, mean or gaussian supported.")
    t0 = time.time()
    # Move to GPU as required, and multiply by sqrt(boxcar_size)
    # This keeps stdev noise the same instead of decreasing by sqrt(N)
    data = cp.asarray(data.astype('float32', copy=False))
    data = uniform_filter1d(data, size=boxcar_size, axis=axis)
    if mode == 'gaussian':
        data *= np.sqrt(boxcar_size)
    elif mode == 'sum':
        data *= boxcar_size
    t1 = time.time()
    logger.info(f"Filter time: {(t1-t0)*1e3:2.2f}ms")

    if return_space == 'cpu':
        return cp.asnumpy(data)
    else:
        return data


def dedoppler(data, metadata, max_dd, min_dd=None, boxcar_size=1,
              boxcar_mode='sum', return_space='cpu', kernel='dedoppler'):
    """ Apply brute-force dedoppler kernel to data

    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space

    Returns:
        dd_vals, dedopp_gpu (np.array, np/cp.array):
    """
    t0 = time.time()
    if min_dd is None:
        min_dd = np.abs(max_dd) * -1

    # Compute minimum possible drift (delta_dd)
    N_time, N_beam, N_chan = data.shape
    if N_beam == 1:
        data = data.squeeze()
    else:
        data = data[:, 0, :] # TODO ADD POL SUPPORT

    obs_len  = N_time * metadata['dt'].to('s').value
    delta_dd = metadata['df'].to('Hz').value / obs_len  # e.g. 2.79 Hz / 300 s = 0.0093 Hz/s

    # Compute dedoppler shift schedules
    N_dopp_upper   = int(max_dd / delta_dd)
    N_dopp_lower   = int(min_dd / delta_dd)

    if max_dd == 0 and min_dd is None:
        dd_shifts = np.array([0], dtype='int32')
    elif N_dopp_upper > N_dopp_lower:
        dd_shifts      = np.arange(N_dopp_lower, N_dopp_upper + 1, dtype='int32')
    else:
        dd_shifts      = np.arange(N_dopp_upper, N_dopp_lower + 1, dtype='int32')[::-1]

    dd_shifts_gpu  = cp.asarray(dd_shifts)
    N_dopp = len(dd_shifts)

    # Copy data over to GPU
    d_gpu = cp.asarray(data.astype('float32', copy=False))

    # Apply boxcar filter
    if boxcar_size > 1:
        d_gpu = apply_boxcar(d_gpu, boxcar_size, mode='sum', return_space='gpu')

    # Allocate GPU memory for dedoppler data
    dedopp_gpu = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
    t1 = time.time()
    logger.info(f"Dedopp setup time: {(t1-t0)*1e3:2.2f}ms")

    # Launch kernel
    t0 = time.time()

    # Setup grid and block dimensions
    F_block = np.min((N_chan, 1024))
    F_grid  = N_chan // F_block
    #print(dd_shifts)
    logger.debug(f"Kernel shape (grid, block) {(F_grid, N_dopp), (F_block,)}")
    if kernel == 'dedoppler':
        dedoppler_kernel((F_grid, N_dopp), (F_block,),
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time)) # grid, block and arguments

    elif kernel == 'kurtosis':
         # output must be scaled by N_acc, which can be figured out from df and dt metadata
        samps_per_sec = (1.0 / np.abs(metadata['df'])).to('s') / 2 # Nyq sample rate for channel
        N_acc = int(metadata['dt'].to('s') / samps_per_sec)
        logger.debug(f'rescaling SK by {N_acc}')
        logger.debug(f"driftrates: {dd_shifts}")
        dedoppler_kurtosis_kernel((F_grid, N_dopp), (F_block,),
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time, N_acc)) # grid, block and arguments

    t1 = time.time()
    logger.info(f"Dedopp kernel time: {(t1-t0)*1e3:2.2f}ms")

    # Compute drift rate values in Hz/s corresponding to dedopp axis=0
    dd_vals = dd_shifts * delta_dd

    metadata['drift_trials'] = dd_vals
    metadata['boxcar_size'] = boxcar_size
    metadata['dd'] = delta_dd * u.Hz / u.s

    # Copy back to CPU if requested
    if return_space == 'cpu':
        logger.info("Dedoppler: copying over to CPU")
        dedopp_cpu = cp.asnumpy(dedopp_gpu)
        return dedopp_cpu, metadata
    else:
        return dedopp_gpu, metadata


def spectral_kurtosis(data, metadata, boxcar_size=1, return_space='cpu'):
    """ Compute spectral kurtosis for zero-drift data """
    dedopp_SK, metadata = dedoppler(data, metadata, boxcar_size=boxcar_size, return_space=return_space,
                                              kernel='kurtosis', max_dd=0)
    return dedopp_SK[0]


def sk_flag(data, metadata, boxcar_size=1, n_sigma_upper=3, n_sigma_lower=2,
            flag_upper=True, flag_lower=True, return_space='cpu'):
    """ Apply spectral kurtosis flagging

    Args:
        data (np.array): Numpy array with shape (N_timestep, N_beam, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        n_sigma_upper (float): Number of stdev above SK estimate to flag (upper bound)
        n_sigma_lower (float): Number of stdev below SK estmate to flag (lower bound)
        flag_upper (bool): Flag channels with large SK (highly variable signals)
        flag_lower (bool): Flag channels with small SK (very stable signals)
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space

    Returns:
        mask (np.array, bool): Array of True/False flags per channel
    """
    samps_per_sec = (1.0 / metadata['df']).to('s') / 2 # Nyq sample rate for channel
    N_acc = int(metadata['dt'].to('s') / samps_per_sec)
    var_theoretical = 2.0 / np.sqrt(N_acc)
    std_theoretical = np.sqrt(var_theoretical)
    sk = spectral_kurtosis(data, metadata, boxcar_size=boxcar_size, return_space=return_space)

    if flag_upper and flag_lower:
        mask  = sk > 1.0 + n_sigma_upper * std_theoretical
        mask  |= sk < 1.0 - (n_sigma_lower * std_theoretical)
    elif flag_upper and not flag_lower:
        mask  = sk > 1.0 + n_sigma_upper * std_theoretical
    elif flag_lower and not flag_upper:
        mask  = sk < 1.0 - (n_sigma_lower * std_theoretical)
    else:
        raise RuntimeError("No flags to process: need to flag upper and/or lower!")
    return mask

    if return_space == 'cpu':
        logger.info("sk_flag: copying over to CPU")
        mask_cpu = cp.asnumpy(mask)
        return mask_cpu
    else:
        return mask


def create_empty_hits_table():
    """ Create empty pandas dataframe for hit data

    Notes:
        Columns are:
            Driftrate (float64): Drift rate in Hz/s
            f_start (float64): Frequency in MHz at start time
            snr (float64): Signal to noise ratio for detection.
            driftrate_idx (int): Index of array corresponding to driftrate
            channel_idx (int): Index of frequency channel for f_start
            boxcar_size (int): Size of boxcar applied to data

    Returns:
        hits (pd.DataFrame): Data frame with columns as above.
    """
    # Create empty dataframe
    hits = pd.DataFrame({'driftrate': pd.Series([], dtype='float64'),
                          'f_start': pd.Series([], dtype='float64'),
                          'snr': pd.Series([], dtype='float64'),
                          'driftrate_idx': pd.Series([], dtype='int'),
                          'channel_idx': pd.Series([], dtype='int'),
                          'boxcar_size': pd.Series([], dtype='int'),
                         })
    return hits


def hitsearch(dedopp, metadata, threshold=10, min_fdistance=None, min_ddistance=None):
    """ Search for hits using _prominent_peaks method in cupyimg.skimage

    Args:
        dedopp (np.array): Dedoppler search array of shape (N_trial, N_chan)
        drift_trials (np.array): List of dedoppler trials corresponding to dedopp N_trial axis
        metadata (dict): Dictionary of metadata needed to convert from indexes to frequencies etc
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        min_ddistance (int): Minimum distance in pixels to nearest peak along doppler axis

    Returns:
        results (pd.DataFrame): Pandas dataframe of results, with columns
                                    driftrate: Drift rate in hz/s
                                    f_start: Start frequency channel
                                    snr: signal to noise ratio
                                    driftrate_idx: Index in driftrate array
                                    channel_idx: Index in frequency array
    """

    drift_trials = metadata['drift_trials']

    if min_fdistance is None:
        min_fdistance = metadata['boxcar_size'] * 2

    if min_ddistance is None:
        min_ddistance = len(drift_trials) // 4

    # Copy over to GPU if required
    dedopp_gpu = cp.asarray(dedopp.astype('float32', copy=False))

    t0 = time.time()
    intensity, fcoords, dcoords = prominent_peaks_optimized(dedopp_gpu, min_xdistance=min_fdistance, min_ydistance=min_ddistance, threshold=threshold)
    logger.debug("# of intensities:{}".format(len(intensity)))

    t1 = time.time()
    logger.info(f"Peak find time: {(t1-t0)*1e3:2.2f}ms")
    t0 = time.time()
    # copy results over to CPU space
    intensity, fcoords, dcoords = cp.asnumpy(intensity), cp.asnumpy(fcoords), cp.asnumpy(dcoords)
    t1 = time.time()
    logger.info(f"Peak find memcopy: {(t1-t0)*1e3:2.2f}ms")

    t0 = time.time()
    if len(fcoords) > 0:
        driftrate_peaks = drift_trials[dcoords]
        logger.debug(f"{metadata['fch1']}, {metadata['df']}, {fcoords}")
        frequency_peaks = metadata['fch1'] + metadata['df'] * fcoords


        results = {
            'driftrate': driftrate_peaks,
            'f_start': frequency_peaks,
            'snr': intensity,
            'driftrate_idx': dcoords,
            'channel_idx': fcoords
        }

        # Append numerical metadata keys
        for key, val in metadata.items():
            if isinstance(val, (int, float)):
                results[key] = val

        return pd.DataFrame(results)
        t1 = time.time()
        logger.info(f"Peak find to dataframe: {(t1-t0)*1e3:2.2f}ms")
    else:
        return None
    
# def populate_domain(hitlist, domain_shape=None):
#     hitlist = hitlist.sort_values("driftrate_idx", ascending=True)
#     split_arr = hitlist["channel_idx"]
#     splitter = np.expand_dims(np.arange(32) * (2**8), axis=0)
#     x = np.expand_dims(np.array(split_arr), axis=1)
#     inds = ((x > splitter-32) * (x < (splitter + (2**8)+32)))
#     sub_lists = [hitlist[sub_inds] for sub_inds in inds.T]
    
#     snr_gpu = cp.zeros(domain_shape, cp.float32)
#     size_gpu = cp.zeros(domain_shape, cp.int8)
    
#     for row in hitlist:
        
        
    
def merge_hits_gpu(hitlist, domain_shape):
    n_drift, n_chan = domain_shape
    F_block = min(n_chan, 2**8)
    n_threads = n_chan // F_block
    
    
    hitlist = hitlist.sort_values("channel_idx", ascending=True)
    split_arr = hitlist["channel_idx"]
    splitter = np.expand_dims(np.arange(n_threads) * F_block, axis=0)
    x = np.expand_dims(np.array(split_arr), axis=1)
    inds = ((x > splitter - 64) * (x < (splitter + F_block + 64)))
    selector = (((np.repeat(np.expand_dims(np.arange(len(hitlist)), axis=1), n_threads, axis=1))) * inds)
    
    ends = np.argmax(selector, axis=0)+1
    selector[inds != True] += len(hitlist)
    starts = np.argmin(selector, axis=0)
    
    snrs = cp.array(hitlist["snr"], dtype=cp.float32)
    drifts = cp.array(hitlist["driftrate_idx"], dtype=cp.int32)
    channels = cp.array(hitlist["channel_idx"], dtype=cp.int32)
    sizes = cp.array(hitlist["boxcar_size"], dtype=cp.int32)
    starts = cp.array(starts, dtype=cp.int32)
    ends = cp.array(ends, dtype=cp.int32)
    save = cp.full((len(hitlist),), True, dtype=cp.bool)
    debug = cp.zeros(len(hitlist), dtype=cp.float32)
    
    NUM_BLOCKS = (1,)
    THREADS_PER_BLOCK = (n_threads,)
    
    suppression_kernel(NUM_BLOCKS, THREADS_PER_BLOCK, (snrs, drifts, channels, sizes, starts, ends, save, debug))
    
    return hitlist[save.get()]
    
    
def merge_hits_orig(hitlist, domain_shape=None):
    """ Group hits corresponding to different boxcar widths and return hit with max SNR 
    Args:
        hitlist (pd.DataFrame): List of hits

    Returns:
        hitlist (pd.DataFrame): Abridged list of hits after merging
    """
    p = hitlist.sort_values('snr', ascending=False)
    hits = []
    while len(p) > 1:
        # Grab top hit
        p0 = p.iloc[0]

        # Find channels and driftrates within tolerances
        q = f"""(abs(driftrate_idx - {p0['driftrate_idx']}) <= boxcar_size + 1  |
                abs(driftrate_idx - {p0['driftrate_idx']}) <= {p0['boxcar_size']} + 1)
                &
                (abs(channel_idx - {p0['channel_idx']}) <= {p0['boxcar_size']} + 1|
                abs(channel_idx - {p0['channel_idx']}) <= boxcar_size + 1)"""
        q = q.replace('\n', '') # Query must be one line
        pq = p.query(q)

        # Drop all matched rows
        p = p.drop(pq.index)
        hits.append(p0)

    return pd.DataFrame(hits)
    
    
def merge_hits_cpu(hitlist, domain_shape=None):
    NUM_CORES = 32
    block_size = int(domain_shape[1] / NUM_CORES)
    split_arr = hitlist["channel_idx"]
    
    splitter = np.expand_dims(np.arange(NUM_CORES) * block_size, axis=0)
    x = np.expand_dims(np.array(split_arr), axis=1)
    inds = ((x > splitter - 64) * (x < (splitter + block_size + 32)))
    sub_lists = [hitlist[sub_inds] for sub_inds in inds.T]
    with Pool(NUM_CORES) as p:
        results = p.map(merge_hits_orig, sub_lists)
    return pd.concat(results).drop_duplicates()

merge_hits = merge_hits_gpu

def run_pipeline(data, metadata, max_dd, min_dd=None, threshold=50, min_fdistance=None,
                 min_ddistance=None, n_boxcar=6, merge_boxcar_trials=True, apply_normalization=False, plot=True):
    """ Run dedoppler + hitsearch pipeline

    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution), as astropy quantities
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        n_boxcar (int): Number of boxcar trials to do, width 2^N e.g. trials=(1,2,4,8,16)
        merge_boxcar_trials (bool): Merge hits of boxcar trials to remove 'duplicates'. Default True.
        apply_normalization (bool): Normalize input data. Default True. Required True for S/N calcs.
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        min_ddistance (int): Minimum distance in pixels to nearest peak along doppler axis
        plot (bool): Whether to plot results

    Returns:
        (dedopp, metadata, peaks): Array of data post dedoppler (at final boxcar width), plus
                                   metadata (dict) and then table of hits (pd.Dataframe).
    """

    t0 = time.time()
    logger.debug(data.shape)
    N_timesteps = data.shape[0]
    _threshold = threshold * np.sqrt(N_timesteps)

    # Apply preprocessing normalization
    if apply_normalization:
        data = normalize(data, return_space='gpu')


    peaks = create_empty_hits_table()

    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    for boxcar_size in boxcar_trials:
        logger.info(f"--- Boxcar size: {boxcar_size} ---")
        dedopp, metadata = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum',
                                     max_dd=max_dd, min_dd=min_dd, return_space='gpu')

        # Plotting original + dedop
        if plot:
            logger.info("Plotting {}".format(boxcar_size))
            plt.figure(figsize=(20, 4))
            plt.subplot(1,2,1)
            imshow_waterfall(data[:,0,:], metadata, 'channel', 'timestep')
            plt.subplot(1,2,2)
            imshow_dedopp(dedopp, metadata, 'channel', 'driftrate')

        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        _threshold = threshold * np.sqrt(N_timesteps * boxcar_size)
        _peaks = hitsearch(dedopp, metadata, threshold=_threshold, min_fdistance=min_fdistance, min_ddistance=min_ddistance)

        # Plotting hits
        if plot:
            overlay_hits(_peaks, 'channel', 'driftrate')
            plt.savefig(f'boxcar: {boxcar_size}.png')

        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
            peaks = pd.concat((peaks, _peaks), ignore_index=True)

    if merge_boxcar_trials:
        t0_merge = time.time()
        peaks = merge_hits(peaks, domain_shape=dedopp.shape)
        t1_merge = time.time()
        logger.info(f"Hit merging time: {(t1_merge-t0_merge)*1e3:2.2f}ms")
    t1 = time.time()

    logger.info(f"Pipeline runtime: {(t1-t0):2.2f}s")
    return dedopp, metadata, peaks


def find_et_serial(filename, filename_out='hits.csv', gulp_size=2**19, max_dd=1, ngulps=0, freq_start=0, *args, **kwargs):
    """ Find ET, serial version

    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.

    Args:
        filename (str): Name of input HDF5 file.
        filename_out (str): Name of output CSV file.
        gulp_size (int): Number of channels to process at once (e.g. N_chan in a coarse channel)
        ngulps (int): Number of gulps to process. If 0, process until EOF.
        freq_start (int): The channel at which to start processing.

    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.

    Notes:
        Passes keyword arguments on to run_pipeline(). Same as find_et but doesn't use dask parallelization.
    """
    t0 = time.time()
    ds = from_h5(filename)
    out = []
    
    i = 0
    while True:
        # Check if we processed enough number of gulps
        if ngulps != 0 and i >= ngulps:
            break
        d_arr = ds.isel({'frequency': slice(freq_start + gulp_size * i, freq_start + gulp_size * (i + 1))})
        d = d_arr.data
        # Check if we ran out of data
        if np.sum(d.shape) == 0:
            break
        i += 1
        f = d_arr.frequency
        t = d_arr.time
        md = {'fch1': f.val_start * f.units, 'df': f.val_step * f.units, 'dt': t.val_step * t.units}
        dedopp, metadata, hits = run_pipeline(d, md, max_dd, *args, **kwargs)
        out.append(hits)
        logger.info(f"{len(hits)} hits found")

    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"## TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe
