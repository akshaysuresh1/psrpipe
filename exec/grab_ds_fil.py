#!/usr/bin/env python
'''
Grab a time chunk of data from a filterbank file and perform the following tasks.

1. Read raw data from a filterbank file.
2. Read an rfifind mask and apply it to the raw data.
3. Compute (if required) and remove bandpass.
4. Remove DM = 0 pc/cc signal (if specified).
5. Downsample (or smooth) data along frequency and time.
'''
from __future__ import print_function
from __future__ import absolute_import
# Load custom modules.
from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.io.read_rfifindmask import read_rfimask, modify_zapchans_bandpass
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, calc_median_bandpass, correct_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.plotting.ds_plot import plot_ds
from psrdynspec.plotting.config import *
# Load standard packages.
from blimpy import Waterfall
from blimpy.io.sigproc import generate_sigproc_header
import numpy as np
import os, time, sys, glob
from argparse import ArgumentParser
#########################################################################
# Run data processing.
def myexecute(hotpotato):
    print('Reading header of file: %s'% (hotpotato['fil_file']))
    hdr = Header(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'],file_type='filterbank') # Returns a Header object
    tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
    t_samp  = hdr.t_samp   # Sampling time (s)
    chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
    nchans  = hdr.nchans   # No. of channels
    npol    = hdr.npol     # No. of polarizations
    n_bytes = hdr.primary['nbits']/8.0 # No. of bytes per data sample
    hdr_size = hdr.primary['hdr_size'] # Header size (bytes)
    times = np.arange(tot_time_samples)*t_samp # 1D array of times (s)
    # Set up frequency array. Frequencies in GHz.
    freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3
    print(hdr)

    # Slice time axis according to the start and end times specified.
    ind_time_low = np.where(times<=hotpotato['start_time'])[0][-1]
    ind_time_high = np.where(times<=hotpotato['end_time'])[0][-1]+1
    times = times[ind_time_low:ind_time_high]

    # Read in a chunk of filterbank data.
    print('Reading in data.')
    f = open(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'],'rb')
    current_cursor_position = f.tell()
    data = load_fil_data(f, ind_time_low, ind_time_high, npol, nchans, n_bytes, hdr_size, hotpotato['pol'], current_cursor_position)

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

    # Chop bandpass edges.
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
        # Set up list of channels to mask in downsampled data.
        mask_zap_check = list(np.sort(mask_zap_chans)//hotpotato['kernel_size_freq_chans'])
        mask_chans = np.array([chan for chan in np.unique(mask_zap_check) if mask_zap_check.count(chan)==hotpotato['kernel_size_freq_chans']])
    else:
        mask_chans = None

    # Remove zerodm signal.
    if hotpotato['remove_zerodm']:
        data = remove_additive_time_noise(data)[0]

    # Smooth and/or downsample the data.
    data, freqs_GHz, times = smooth_master(data,hotpotato['smoothing_method'],hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times)
    if hotpotato['smoothing_method']!='Blockavg2D':
        data, freqs_GHz, times = smooth_master(data,'Blockavg2D',hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times)

    # Remove residual spectral trend.
    print('Removing residual spectral trend')
    data = data - np.median(data,axis=1)[:,None]

    # Remove any residual temporal trend.
    if hotpotato['remove_zerodm']:
        data = data - np.median(data, axis=0)[None,:]
        print('Zerodm removal completed.')

    if mask_chans is not None:
        data[mask_chans] = 0.0

    # Clip off masked channels at edges of the frequency band.
    if mask_chans is not None:
        # Lowest channel not to be masked.
        low_ch_index = 0
        while low_ch_index+1 in mask_chans:
            low_ch_index += 1
        # Highest channel not to be masked.
        high_ch_index = len(freqs_GHz)-1
        while high_ch_index in mask_chans:
            high_ch_index -= 1
        freqs_GHz = freqs_GHz[low_ch_index:high_ch_index+1]
        data = data[low_ch_index:high_ch_index+1]
        # Modify channel mask to reflect properties of updated data range.
        mask_chans = np.delete(mask_chans, np.where(mask_chans<low_ch_index))
        mask_chans = np.delete(mask_chans, np.where(mask_chans>high_ch_index))
        mask_chans = np.array(mask_chans - low_ch_index, dtype=int)

    # Produce imshow plot of data.
    if not os.path.isdir(hotpotato['OUTPUT_DIR']):
        os.makedirs(hotpotato['OUTPUT_DIR'])

    plot_ds(data,times,freqs_GHz,hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename'],show_plot=hotpotato['show_plot'],time_unit='s',freq_unit='GHz',flux_unit='arbitrary units',vmin=np.mean(data)-2*np.std(data),vmax=np.mean(data)+5*np.std(data),log_colorbar=False,cmap=hotpotato['cmap'],mask_chans=mask_chans)

    # Update header to reflect data properties.
    hdr.primary.pop('hdr_size', None)
    hdr.primary['fch1'] = freqs_GHz[0]*1e3
    hdr.primary['foff'] = (freqs_GHz[1]-freqs_GHz[0])*1e3
    hdr.primary['nchans'] = len(freqs_GHz)
    hdr.primary['nifs'] = 1
    hdr.primary['tsamp'] = times[1] - times[0]
    hdr.primary['nbits'] = 32 # Cast data to np.float32 type.

    # Write data to either .npz file or a filterbank file.
    if hotpotato['do_write']:
        if hotpotato['write_format']=='npz':
            write_npz(data, freqs_GHz, times, mask_chans, hotpotato)
        elif hotpotato['write_format']=='fil' or hotpotato['write_format']=='filterbank':
            write_fil(data, times, freqs_GHz, hdr, hotpotato)
        else:
            print('File write format not recognized. Terminating program execution.')

    return data, freqs_GHz, times

# Write data priducts to npz file.
def write_npz(data, freqs_GHz, times, mask_chans, hotpotato):
    npz_filename = hotpotato['OUTPUT_DIR'] + '/' + hotpotato['basename'] + '_t%.2fto%.2f_freqs%.2fto%.2f'% (times[0], times[-1], freqs_GHz[0], freqs_GHz[-1])
    save_array = [data, freqs_GHz, times, mask_chans]
    save_keywords = ['DS', 'Radio frequency (GHz)', 'Time (s)', 'Channel mask']
    np.savez(npz_filename,**{name:value for name, value in zip(save_keywords, save_array)})

# Write data products to filterbank format.
def write_fil(data, times, freqs_GHz, hdr, hotpotato):
    # Reshape data array.
    data = data.T.reshape((data.T.shape[0], 1, data.T.shape[1])) # New shape = (No. of time samples, No. of polarizations, No. of channels)
    # Construct a Waterfall object that will be written to disk as a filterbank file.
    base_fileout = hotpotato['basename']+'_t%.2fto%.2f_freqs%.2fto%.2f'% (times[0], times[-1], freqs_GHz[0], freqs_GHz[-1]) +'.fil'
    filename_out = hotpotato['OUTPUT_DIR']+'/'+base_fileout
    wat = Waterfall() # Empty Waterfall object
    wat.header = hdr.primary
    with open(filename_out, 'wb') as fh:
        print('Writing smoothed data to %s'% (base_fileout))
        fh.write(generate_sigproc_header(wat)) # Trick Blimpy into writing a sigproc header.
        np.float32(data.ravel()).tofile(fh)

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['show_plot']=='':
        hotpotato['show_plot'] = False
    if hotpotato['do_write']=='':
        hotpotato['do_write'] = False
    if hotpotato['write_format']=='':
        hotpotato['write_format'] = 'npz'
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
    """ Command line tool for plotting dynamic spectrum. """
    parser = ArgumentParser(description="Process and plot dynamic spectrum chunks from one or more data sets.")
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
    data, freqs_GHz, times = myexecute(hotpotato)

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    print('Code run time = %.5f minutes'% (run_time))

    return data, freqs_GHz, times, hotpotato
#########################################################################
if __name__=='__main__':
    data, freqs_GHz, times, hotpotato = main()
#########################################################################
