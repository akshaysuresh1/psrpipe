'''
This scripts performs the following tasks.
1. Read raw data from a collection of PSRFITS files.
2. Read an rfifind mask and apply it to the raw data.
3. Compute (if required) and remove bandpass.
4. Remove DM = 0 pc/cc signal (if specified).
5. Downsample (or smooth) data along frequency and time.
'''
# Load custom modules.
from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.io.read_psrfits import load_psrfits_data
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, calc_median_bandpass, correct_bandpass
from psrdynspec.io.read_rfifindmask import read_rfimask, modify_zapchans_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
# Load standard packages.
import numpy as np
import os, time, glob
#########################################################################
# Profile code execution.
prog_start_time = time.time() # Start time
############################################################################
# INPUTS
dict = read_config('mask_ds_psrfits.cfg')

# Set deafults.
if dict['OUTPUT_DIR']=='':
    dict['OUTPUT_DIR'] = dict['DATA_DIR']
if dict['pol']=='':
    dict['pol'] = 0
if dict['RFIMASK_DIR']=='':
    dict['RFIMASK_DIR'] = dict['DATA_DIR']
if dict['BANDPASS_DIR']=='':
    dict['BANDPASS_DIR'] = dict['DATA_DIR']
if dict['remove_zerodm']=='':
    dict['remove_zerodm'] = False
############################################################################
# Reading filterbank header info.
data_file = dict['DATA_DIR']+'/'+dict['psrfits_file']
print('Reading header of file: ',dict['psrfits_file'])
hdr = Header(data_file,file_type='psrfits') # Returns a Header object
tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
t_samp  = hdr.t_samp   # Sampling time (s)
chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
nchans  = hdr.nchans   # No. of channels
npol    = hdr.npol     # No. of polarizations
times = np.arange(tot_time_samples)*t_samp # 1D array of times (s)
# Set up frequency array. Frequencies in GHz.
freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3
# Flip frequency axis if chan_bw<0.
if (chan_bw<0):
    freqs_GHz = np.flip(freqs_GHz)
    print('Frequencies arranged in ascending order.')

# Load data.
print('Reading in data.')
data = load_psrfits_data(dict['psrfits_file'],dict['DATA_DIR']+'/',dict['pol'])
# Flip frequency axis of DS if channel bandwidth is negative.
if (chan_bw<0):
    print('Flipping frequency axis of DS')
    data = np.flip(data,axis=0)

if dict['bandpass_method']=='compute':
    print('Computing median bandpass')
    median_bp = calc_median_bandpass(data)
    print('Median bandpass computed.')
elif dict['bandpass_method']=='file':
    print('Loading median bandpass from %s'% (dict['bandpass_npz']))
    bp_contents = np.load(dict['BANDPASS_DIR']+dict['bandpass_npz'],allow_pickle=True)
    median_bp = bp_contents['Median bandpass']
    print('Median bandpass loaded.')
else:
    print('Bandpass computation method not recognized.')

# Discard bandpass edges.
print('Clipping bandpass edges.')
ind_band_low = np.where(freqs_GHz>=dict['freq_band_low'])[0][0]
ind_band_high = np.where(freqs_GHz<=dict['freq_band_high'])[0][-1]+1
freqs_GHz = freqs_GHz[ind_band_low:ind_band_high]
median_bp = median_bp[ind_band_low:ind_band_high]
data = data[ind_band_low:ind_band_high]

# Correct bandpass.
print('Correct bandpass shape')
if 0 in median_bp:
    indices_zero_bp = np.where(median_bp==0)[0]
    replace_value = np.median(median_bp[np.where(median_bp!=0)[0]])
    median_bp[indices_zero_bp] = replace_value
    data[indices_zero_bp] = replace_value
data = correct_bandpass(data, median_bp)

# Read in rfifind mask.
print('Reading in rfifind mask: %s'% (dict['rfimask']))
nint, int_times, ptsperint, mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int = read_rfimask(dict['RFIMASK_DIR']+'/'+dict['rfimask'])
mask_zap_chans, mask_zap_chans_per_int = modify_zapchans_bandpass(mask_zap_chans, mask_zap_chans_per_int, ind_band_low, ind_band_high)
boolean_rfimask = np.zeros(data.shape,dtype=bool)
for i in range(nint):
    if i==nint-1:
        tstop_int = tot_time_samples
    else:
        tstop_int = np.min(np.where(times>=int_times[i+1])[0])
    tstart_int = np.min(np.where(times>=int_times[i])[0])
    boolean_rfimask[mask_zap_chans_per_int[i],tstart_int:tstop_int] = True
print('Applying RFI mask on data')
data = np.ma.MaskedArray(data,mask=boolean_rfimask)
# Replaced masked entried with median value.
print('Replacing masked entries with median values')
data = np.ma.filled(data, fill_value=np.nanmedian(data))

# Zerodm removal
if dict['remove_zerodm']:
    print('Perfoming zerodm removal')
    data = remove_additive_time_noise(data)[0]

# Smooth and downsample the data.
data, freqs_GHz, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],freqs_GHz,times)

import matplotlib.pyplot as plt
tstart_index = 0
tstop_index = 5000
rms_data = np.std(data)
plt.imshow(data[:,tstart_index:tstop_index+1], aspect='auto', interpolation='nearest', origin='lower', extent=[times[tstart_index], times[tstop_index], freqs_GHz[0], freqs_GHz[-1]],vmax=5*rms_data, vmin=-3*rms_data)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Radio frequency (GHz)', fontsize=14)
h = plt.colorbar()
h.set_label('Flux density (arbitrary units)', fontsize=14)
plt.show()
#########################################################################
# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE
#########################################################################
