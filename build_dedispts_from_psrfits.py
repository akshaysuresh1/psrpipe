# Build a dedispersed dynamic spectrum and time series from a set of PSRFITS files.

from psrdynspec import read_config, Header, load_psrfits_data
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, correct_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.modules.dedisperse import dedisperse_ds
from psrdynspec.plotting.config import *
from psrdynspec.plotting.dedisperse_plot import plot_dedispersed_ds
from psrdynspec.plotting.bandpass_plot import plot_bandpass
from psrdynspec.plotting.ds_plot import plot_ds

import os, time
import numpy as np
############################################################################
# INPUTS
dict = read_config('build_dedispts_from_psrfits.cfg')

# Set default values for empty dictionary items.
if (dict['BANDPASS_DIR']==''):
    dict['BANDPASS_DIR'] = dict['OUTPUT_DIR']
if (dict['DS_SAVE_DIR']==''):
    dict['DS_SAVE_DIR'] = dict['OUTPUT_DIR']
if (dict['show_plot']==''):
    dict['show_plot'] = False
if (dict['log_colorbar_ds']==''):
    dict['log_colorbar_ds'] = False
if (dict['vmin_percentile']==''):
    dict['vmin_percentile'] = None
if (dict['vmax_percentile']==''):
    dict['vmax_percentile'] = None
if (dict['DM']==''):
    dict['DM'] = 0.0
if (dict['pol']==''):
    dict['pol'] = 0
if (dict['convolution_method']==''):
    dict['convolution_method'] = 'fftconvolve'
############################################################################
# Units for plots
time_unit = 's'
freq_unit = 'GHz'
flux_unit = 'arb. units'
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['BANDPASS_DIR']):
    os.makedirs(dict['BANDPASS_DIR'])
if not os.path.isdir(dict['DS_SAVE_DIR']):
    os.makedirs(dict['DS_SAVE_DIR'])
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time()

# Read header and extract metadata.
hdr = Header(dict['DATA_DIR']+dict['glob_fits_files'],file_type='psrfits')
tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
t_samp  = hdr.t_samp   # Sampling time (s)
chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
nchans  = hdr.nchans   # No. of channels
npol    = hdr.npol     # No. of polarizations
times = np.arange(tot_time_samples)*t_samp # 1D array of times (s)
freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3 # 1D array of radio frequencies (GHz)

# Load data.
data = load_psrfits_data(dict['glob_fits_files'],dict['DATA_DIR'],dict['pol'])

# Flip frequency axis if chan_bw <0.
if (chan_bw<0):
    freqs_GHz = np.flip(freqs_GHz)
    data = np.flip(data,axis=0)
    print('Frequencies arranged in ascending order.')

# Discard data beyond specified band edges.
print('Discarding bandpass edges...')
ind_band_low = np.where(freqs_GHz>=dict['freq_band_low'])[0][0]
ind_band_high = np.where(freqs_GHz<=dict['freq_band_high'])[0][-1]
freqs_GHz = freqs_GHz[ind_band_low:ind_band_high+1]
data = data[ind_band_low:ind_band_high+1]

# Calculate median bandpass.
print('Calculating median bandpass...')
median_bp = np.zeros(len(data))
for i in range(len(data)):
    median_bp[i] = np.median(data[i])
print('Calculation completed.')
plot_bandpass(freqs_GHz,median_bp,freq_unit,flux_unit,dict['basename'],dict['BANDPASS_DIR'],dict['show_plot'])

# Correct data for bandpass shape.
data = correct_bandpass(data, median_bp)
# Smooth the dynamic spectrum using specified paramters.
data, freqs_GHz, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],freqs_GHz,times)
t_samp_smoothed = times[1] - times[0]
# vmin
if (dict['vmin_percentile']!=None):
    vmin = np.nanpercentile(data,dict['vmin_percentile'])
else:
    vmin = None
# vmax
if (dict['vmax_percentile']!=None):
    vmax = np.nanpercentile(data,dict['vmax_percentile'])
else:
    vmax = None
plot_ds(data,times[0],times[-1],freqs_GHz[0],freqs_GHz[-1],time_unit,freq_unit,flux_unit,dict['basename'],dict['DS_SAVE_DIR'],dict['show_plot'],vmin=vmin,vmax=vmax,log_colorbar=dict['log_colorbar_ds'])

# Reference frequency and low frequency limit for dedispersion.
ref_freq = np.max(freqs_GHz) # Top of the highest subband
freq_low = np.min(freqs_GHz) # Bottom of the lowest subband

# Dedisperse the whole dynamic spectrum at specified DM.
start_time = times[0]
print('Dedispersing dynamic spectrum at DM = %.2f pc/cc'% (dict['DM']))
dedisp_ds, dedisp_times, dedisp_ts = dedisperse_ds(data,freqs_GHz,dict['DM'],ref_freq,freq_low,t_samp_smoothed,start_time)
print('Dedispersion complete.')
plot_dedispersed_ds(dedisp_ds,dedisp_ts,dedisp_times,freqs_GHz,start_time,t_samp_smoothed,dict['DM'],time_unit,freq_unit,flux_unit,dict['basename'],dict['DS_SAVE_DIR'],dict['show_plot'],vmin=vmin,vmax=vmax,log_colorbar=dict['log_colorbar_ds'])

# Output file name
output_file_basename = dict['basename']+'_DM'+'%0.1f'% (dict['DM'])
# Save data products to NumPy file
save_array = [dedisp_times,dedisp_ts,dedisp_ds,freqs_GHz]
save_keywords = ['Time (s)','Dedisp time series','Dedisp DS','Frequencies (GHz)']
np.save(dict['OUTPUT_DIR']+output_file_basename+'.npy',save_array)

# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
