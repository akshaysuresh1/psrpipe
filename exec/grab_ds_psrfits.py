#!/usr/bin/env python
'''
Grab a time chunk of data from a collection of PSRFITS files and perform the following tasks.

1. Read raw data from a collection of PSRFITS files.
2. Read an rfifind mask and apply it to the raw data.
3. Compute (if required) and remove bandpass.
4. Remove DM = 0 pc/cc signal (if specified).
5. Downsample (or smooth) data along frequency and time.
'''
from __future__ import print_function
from __future__ import absolute_import
# Load custom modules.
from psrdynspec import read_config
from psrdynspec.io.read_psrfits import extract_psrfits_datachunk
from psrdynspec.io.read_rfifindmask import read_rfimask, modify_zapchans_bandpass
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, calc_median_bandpass, correct_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.plotting.config import *
from psrdynspec.plotting.ds_plot import plot_ds
# Load standard packages.
from blimpy import Waterfall
from blimpy.io.sigproc import generate_sigproc_header
import numpy as np
import os, time, sys, glob
from argparse import ArgumentParser
#########################################################################
# Run data processing.
def myexecute(hotpotato):
    print('Reading file: %s'% (hotpotato['glob_psrfits']))
    data, hdr, st_time = extract_psrfits_datachunk(hotpotato['DATA_DIR']+'/'+hotpotato['glob_psrfits'], hotpotato['start_time'], hotpotato['end_time'], hotpotato['pol'])
    print(hdr)
    tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
    t_samp  = hdr.t_samp   # Sampling time (s)
    chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
    nchans  = hdr.nchans   # No. of channels
    times = st_time+np.arange(data.shape[-1])*t_samp # 1D array of times (s)
    # Set up frequency array. Frequencies in GHz.
    freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3

    # Flip frequency axis if chan_bw<0.
    if (chan_bw<0):
        print('Channel bandwidth is negative.')
        print('Flipping frequency axis of DS')
        data = np.flip(data,axis=0)
        freqs_GHz = np.flip(freqs_GHz)
        print('Frequencies rearranged in ascending order.')

    # Load/compute median bandpass.
    if hotpotato['bandpass_method']=='file':
        print('Loading median bandpass from %s'% (hotpotato['bandpass_npz']))
        median_bp = np.load(hotpotato['BANDPASS_DIR']+'/'+hotpotato['bandpass_npz'],allow_pickle=True)['Median bandpass']
        print('Median bandpass loaded.')
    elif hotpotato['bandpass_method']=='compute':
        print('Computing median bandpass')
        median_bp = calc_median_bandpass(data)
    else:
        print('Unrecognized bandpass computation method. Quitting program..')
        sys.exit(1)

    ind_band_low = np.where(freqs_GHz>=hotpotato['freq_band_low'])[0][0]
    ind_band_high = np.where(freqs_GHz<=hotpotato['freq_band_high'])[0][-1]+1
    # Clip bandpass edges.
    freqs_GHz = freqs_GHz[ind_band_low:ind_band_high]
    data = data[ind_band_low:ind_band_high]
    median_bp = median_bp[ind_band_low:ind_band_high]
    print('Bandpass edges clipped.')

    # Correct bandpass shape.
    print('Correcting data for bandpass shape')
    if 0 in median_bp:
        print('Replacing zeros in bandpass shape with median values')
        indices_zero_bp = np.where(median_bp==0)[0]
        replace_value = np.median(median_bp[np.where(median_bp!=0)[0]])
        median_bp[indices_zero_bp] = replace_value
        data[indices_zero_bp] = replace_value
    data = correct_bandpass(data, median_bp)

    # Remove zerodm signal.
    if hotpotato['remove_zerodm']:
        data = remove_additive_time_noise(data)[0]

    # Read and apply rfifind mask.
    if hotpotato['apply_rfimask']:
        print('Reading rfifind mask %s'% (hotpotato['rfimask']))
        nint, int_times, ptsperint, mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int = read_rfimask(hotpotato['RFIMASK_DIR']+'/'+hotpotato['rfimask'])
        mask_zap_chans, mask_zap_chans_per_int = modify_zapchans_bandpass(mask_zap_chans, mask_zap_chans_per_int, ind_band_low, ind_band_high)
        idx1 = np.where(int_times<=times[0])[0][-1]
        idx2 = np.where(int_times<=times[-1])[0][-1] + 1
        nint = idx2 - idx1
        int_times = int_times[idx1:idx2]
        mask_zap_chans_per_int = mask_zap_chans_per_int[idx1:idx2]
        mask_zap_ints = mask_zap_ints[np.where(np.logical_and(mask_zap_ints>=times[0], mask_zap_ints<=times[-1]))[0]]
        # Apply rfifind mask on data.
        boolean_rfimask = np.zeros(data.shape,dtype=bool)
        for i in range(nint):
            if i==nint-1:
                tstop_int = len(times)
            else:
                tstop_int = np.min(np.where(times>=int_times[i+1])[0])
            tstart_int = np.min(np.where(times>=int_times[i])[0])
            boolean_rfimask[mask_zap_chans_per_int[i],tstart_int:tstop_int] = True
        print('Applying RFI mask on data')
        data = np.ma.MaskedArray(data,mask=boolean_rfimask)
        # Replaced masked entries with mean value.
        print('Replacing masked entries with mean values')
        data = np.ma.filled(data, fill_value=np.nanmean(data))

    # Smooth and/or downsample the data.
    data, freqs_GHz, times = smooth_master(data,hotpotato['smoothing_method'],hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times)
    if hotpotato['smoothing_method']!='Blockavg2D':
        data, freqs_GHz, times = smooth_master(data,'Blockavg2D',hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times)

    # Remove zerodm signal.
    if hotpotato['remove_zerodm']:
        print('Removing zerodm component')
        data = data - np.mean(data,axis=0)

    # Remove residual spectral trend.
    print('Removing residual spectral trend')
    data = data - np.median(data,axis=1)[:,None]

    # Produce imshow plot of data.
    if not os.path.isdir(hotpotato['OUTPUT_DIR']):
        os.makedirs(hotpotato['OUTPUT_DIR'])
    # TO DO: Incorporate labeling of flagged channels in dynamic spectrum plot.        
    plot_ds(data,times[0],times[-1],freqs_GHz[0],freqs_GHz[-1],'s','GHz','arbitrary units',hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename'],show_plot=hotpotato['show_plot'],vmin=-2.0*np.std(data),vmax=5.0*np.std(data),log_colorbar=False,cmap=hotpotato['cmap'])

    # Write dynamic spectrum to disk as .npz file.
    npz_filename = hotpotato['OUTPUT_DIR'] + '/' + hotpotato['basename'] + '_t%.3fto%.3f_freqs%.2fto%.2f'% (times[0], times[-1], freqs_GHz[0], freqs_GHz[-1])
    save_array = [data, freqs_GHz, times, hdr]
    save_keywords = ['DS', 'Radio frequency (GHz)', 'Time (s)', 'Header']
    np.savez(npz_filename,**{name:value for name, value in zip(save_keywords, save_array)})

    return data, freqs_GHz, times, hdr

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['cmap']=='':
        hotpotato['cmap'] = 'viridis'
    if hotpotato['show_plot']=='':
        hotpotato['show_plot'] = False
    if hotpotato['pol']=='':
        hotpotato['pol'] = 0
    if hotpotato['apply_rfimask']=='':
        hotpotato['apply_rfimask'] = False
    if hotpotato['RFIMASK_DIR']=='':
        hotpotato['RFIMASK_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['BANDPASS_DIR']=='':
        hotpotato['BANDPASS_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['remove_zerodm']=='':
        hotpotato['remove_zerodm'] = False
    if hotpotato['smoothing_method']=='':
        hotpotato['smoothing_method'] = 'Blockavg2D'
    if hotpotato['convolution_method']=='':
        hotpotato['convolution_method'] = 'fftconvolve'
    return hotpotato
#########################################################################
def main():
    """ Command line tool for running rfifind. """
    parser = ArgumentParser(description="Process and plot dynamic spectra from one or more PSRFITS files.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg
    hotpotato = set_defaults(read_config(inputs_cfg))

    # Profile code execution.
    prog_start_time = time.time()

    # Run data processing.
    data, freqs_GHz, times, hdr = myexecute(hotpotato)

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    print('Code run time = %.5f minutes'% (run_time))

    return data, freqs_GHz, times, hdr, hotpotato
#########################################################################
if __name__=='__main__':
    data, freqs_GHz, times, hdr, hotpotato = main()
#########################################################################
