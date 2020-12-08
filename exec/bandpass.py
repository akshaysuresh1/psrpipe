#!/usr/bin/env python
'''
Calculate median bandpass using a chunk of data extracted from one or more PSRFITS/filterbank files.
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.io.read_psrfits import extract_psrfits_datachunk
from psrdynspec.modules.ds_systematics import calc_median_bandpass
from psrdynspec.modules.filters1d import savgol_lowpass
from psrdynspec.plotting.config import *
from psrdynspec.plotting.bandpass_plot import plot_bandpass_subbands, plot_bandpass
# Standard imports
import os, time, sys, glob
import numpy as np
from argparse import ArgumentParser
############################################################################
# INPUTS
def set_defaults(hotpotato):
    # Set default values for empty dictionary items.
    if (hotpotato['show_plot']==''):
        hotpotato['show_plot'] = False
    if (hotpotato['start_time']==''):
        hotpotato['start_time'] = 0.0
    if (hotpotato['duration']==''):
        hotpotato['duration'] = None
    if (hotpotato['pol']==''):
        hotpotato['pol'] = 0
    if (hotpotato['freqs_extract']==''):
        hotpotato['freqs_extract'] = None
    else:
        hotpotato['freqs_extract'] = np.array(hotpotato['freqs_extract'])
    if not isinstance(hotpotato['window_length'],int):
        hotpotato['window_length'] = 1
    else:
        hotpotato['window_length'] = (hotpotato['window_length']//2)*2 + 1
    if (hotpotato['poly_degree']==''):
        hotpotato['poly_degree'] = 1
    return hotpotato
############################################################################
def calc_bandpass(inputs_cfg):
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    hotpotato = set_defaults(hotpotato)

    # Read header.
    print('Reading header of file %s'% (hotpotato['data_file']))
    hdr = Header(hotpotato['DATA_DIR']+'/'+hotpotato['data_file'],file_type=hotpotato['file_type'])
    print(hdr)
    tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
    t_samp  = hdr.t_samp   # Sampling time (s)
    chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
    nchans  = hdr.nchans   # No. of channels
    npol    = hdr.npol     # No. of polarizations
    freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3 # 1D array of radio frequencies (GHz)

    # Start and stop indices along the time axis.
    t_start = int(hotpotato['start_time']//t_samp)
    if (hotpotato['duration']== None):
        t_stop = tot_time_samples
    else:
        t_stop = int((hotpotato['start_time'] + hotpotato['duration'])//t_samp)
    times = np.arange(t_start,t_stop)*t_samp

    # Extracting the data.
    print('Reading %s:' %(hotpotato['data_file']))
    if (hotpotato['file_type']=='psrfits'):
        raw_ds, hdr, acc_start_time = extract_psrfits_datachunk(hotpotato['DATA_DIR']+'/'+hotpotato['data_file'], times[0], times[-1], hotpotato['pol'])
    elif (hotpotato['file_type']=='filterbank'):
        f = open(hotpotato['DATA_DIR']+'/'+hotpotato['data_file'],'rb')
        raw_ds = load_fil_data(f,t_start,t_stop,npol,nchans, hdr.primary['nbits']/8.0,hdr.primary['hdr_size'],hotpotato['pol'],f.tell())
        f.close()
    else:
        sys.exit('Unsupported file type: %s'% (hotpotato['file_type']))
    print('Data extracted. Data shape = ',raw_ds.shape)

    # Flip along the frequency axis if channel bandwidth is negative.
    if (chan_bw<0):
        raw_ds = np.flip(raw_ds,axis=0)
        freqs_GHz = np.flip(freqs_GHz)
        print('Flipped frequency axis of dynamic spectrum.')

    # Calculate median bandpass shape.
    print("Calculating median bandpass...")
    median_bp = calc_median_bandpass(raw_ds)
    print('Median bandpass computed.')

    # Create output directory if non-existent.
    if not os.path.isdir(hotpotato['OUTPUT_DIR']):
        os.makedirs(hotpotato['OUTPUT_DIR'])

    # Section the raw DS into subbands as specified.
    if (hotpotato['freqs_extract'] is not None):
        freqs_subband = [] # List to store frequency information for each subband.
        bandpass_fit_subband = [] # List to store calculated bandpass shape for each subband.
        print('Sectioning raw DS into specified subbands..')
        for j in range(len(hotpotato['freqs_extract'])):
            ind_low = np.where(freqs_GHz>=hotpotato['freqs_extract'][j,0])[0][0]
            ind_high = np.where(freqs_GHz>hotpotato['freqs_extract'][j,1])[0][0]
            f_subband = freqs_GHz[ind_low:ind_high] # Frequency array for this subband
            # Smooth the median bandpass with a low-pass filter.
            print('Smoothing median bandpass over the subband: %s - %s GHz'% (hotpotato['freqs_extract'][j,0],hotpotato['freqs_extract'][j,1]))
            smooth_bp = savgol_lowpass(median_bp[ind_low:ind_high],hotpotato['window_length'],hotpotato['poly_degree'])
            # Update lists to reflect extracted subband information.
            freqs_subband.append(f_subband)
            bandpass_fit_subband.append(smooth_bp)
        # Covert lists to arrays.
        freqs_subband = np.array(freqs_subband)
        bandpass_fit_subband = np.array(bandpass_fit_subband)
        # Plot bandpass with color-coded subbands.
        plot_bandpass_subbands(freqs_GHz,median_bp,freqs_subband,bandpass_fit_subband,'GHz','arb. units',hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename'],hotpotato['show_plot'])
    else:
        freqs_subband = None
        bandpass_fit_subband = None
        plot_bandpass(freqs_GHz,median_bp,'GHz','arb. units',hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename'],hotpotato['show_plot'])

    # Save bandpass information to npz file
    file_name = hotpotato['basename']+'_bandpass'
    save_array = [freqs_GHz,median_bp,hotpotato['freqs_extract'],freqs_subband,bandpass_fit_subband,times,hotpotato['window_length'],hotpotato['poly_degree']]
    save_keywords = ['Radio frequency (GHz)','Median bandpass','Band edges (GHz)','Subband Frequency (GHz)','Smooth subband bandpass','Time (s)','Savgol window size','Savgol polynomial deg']
    np.savez(hotpotato['OUTPUT_DIR']+'/'+file_name,**{name:value for name, value in zip(save_keywords,save_array)})

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    print('Code run time = %.2f minutes'% (run_time))
############################################################################
def main():
    """ Command line tool for median bandpass computation. """
    parser = ArgumentParser(description="Calculate median bandpass shape from a PSRFITS / filterbank file.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs to median bandpass computation.")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg

    calc_bandpass(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
