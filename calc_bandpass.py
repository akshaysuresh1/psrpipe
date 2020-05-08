# Calculate median bandpass using a chunk of data extracted from one or more PSRFITS/filterbank files.

from psrdynspec import read_config, Header, load_psrfits_data, load_fil_data
from psrdynspec.modules.ds_systematics import calc_median_bandpass
from psrdynspec.modules.filters1d import savgol_lowpass
from psrdynspec.plotting.config import *
from psrdynspec.plotting.bandpass_plot import plot_bandpass_subbands, plot_bandpass

import os, time, sys
import numpy as np
############################################################################
# INPUTS
dict = read_config('calc_bandpass.cfg')

# Set default values for empty dictionary items.
if (dict['show_plot']==''):
    dict['show_plot'] = False
if (dict['start_time']==''):
    dict['start_time'] = 0.0
if (dict['duration']==''):
    dict['duration'] = None
if (dict['pol']==''):
    dict['pol'] = 0
if (dict['freqs_extract']==''):
    dict['freqs_extract'] = None
else:
    dict['freqs_extract'] = np.array(dict['freqs_extract'])
if not isinstance(dict['window_length'],int):
    dict['window_length'] = 1
else:
    dict['window_length'] = (dict['window_length']//2)*2 + 1
if (dict['poly_degree']==''):
    dict['poly_degree'] = 1
############################################################################
# Units for plots
time_unit = 's'
freq_unit = 'GHz'
flux_unit = 'arb. units'
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time()

# Read header.
print('Reading header of file %s'% (dict['data_file']))
hdr = Header(dict['DATA_DIR']+dict['data_file'],file_type=dict['file_type'])
tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
t_samp  = hdr.t_samp   # Sampling time (s)
chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
nchans  = hdr.nchans   # No. of channels
npol    = hdr.npol     # No. of polarizations
freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3 # 1D array of radio frequencies (GHz)

# Start and stop indices along the time axis.
t_start = int(dict['start_time']//t_samp)
if (dict['duration']== None):
    t_stop = tot_time_samples
else:
    t_stop = int((dict['start_time'] + dict['duration'])//t_samp)
times = np.arange(t_start,t_stop)*t_samp

# Extracting the data.
print('Reading data from %s:' %(dict['data_file']))
if (dict['file_type']=='psrfits'):
    raw_ds = load_psrfits_data(dict['data_file'],dict['DATA_DIR'],dict['pol'])[...,t_start:t_stop]
elif (dict['file_type']=='filterbank'):
    f = open(dict['DATA_DIR']+dict['data_file'],'rb')
    raw_ds = load_fil_data(dict['data_file'],dict['DATA_DIR'],t_start,t_stop,npol,nchans, hdr.primary[b'nbits']/8.0,f,hdr.primary['hdr_size'],dict['pol'],current_cursor_position=0)
    f.close()
else:
    sys.exit('Unsupported file type: %s'% (dict['file_type']))
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

# Section the raw DS into subbands as specified.
if (dict['freqs_extract'] is not None):
    freqs_subband = [] # List to store frequency information for each subband.
    bandpass_fit_subband = [] # List to store calculated bandpass shape for each subband.
    print('Sectioning raw DS into specified subbands..')
    for j in range(len(dict['freqs_extract'])):
        ind_low = np.where(freqs_GHz>=dict['freqs_extract'][j,0])[0][0]
        ind_high = np.where(freqs_GHz>dict['freqs_extract'][j,1])[0][0]
        f_subband = freqs_GHz[ind_low:ind_high] # Frequency array for this subband
        # Smooth the median bandpass with a low-pass filter.
        print('Smoothing median bandpass over the subband: %s - %s GHz'% (dict['freqs_extract'][j,0],dict['freqs_extract'][j,1]))
        smooth_bp = savgol_lowpass(median_bp[ind_low:ind_high],dict['window_length'],dict['poly_degree'])
        # Update lists to reflect extracted subband information.
        freqs_subband.append(f_subband)
        bandpass_fit_subband.append(smooth_bp)

    # Covert lists to arrays.
    freqs_subband = np.array(freqs_subband)
    bandpass_fit_subband = np.array(bandpass_fit_subband)
    # Plot bandpass with color-coded subbands.
    plot_bandpass_subbands(freqs_GHz,median_bp,freqs_subband,bandpass_fit_subband,freq_unit,flux_unit,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'])
else:
    freqs_subband = None
    bandpass_fit_subband = None
    plot_bandpass(freqs_GHz,median_bp,freq_unit,flux_unit,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'])

# Save bandpass information to npz file
file_name = dict['basename']+'_bandpass'
save_array = [freqs_GHz,median_bp,dict['freqs_extract'],freqs_subband,bandpass_fit_subband,times,dict['window_length'],dict['poly_degree']]
save_keywords = ['Radio frequency (GHz)','Median bandpass','Band edges (GHz)','Subband Frequency (GHz)','Smooth subband bandpass','Time (s)','Savgol window size','Savgol polynomial deg']
np.savez(dict['OUTPUT_DIR']+file_name,**{name:value for name, value in zip(save_keywords,save_array)})

# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
