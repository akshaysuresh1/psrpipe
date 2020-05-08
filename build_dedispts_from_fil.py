# Construct a dedispersed time series at a specific DM from a single filterbank file.

from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, correct_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.modules.dedisperse import calc_block_length, calc_tDM, dedisperse_ds

import os, time
import numpy as np
############################################################################
# INPUTS
dict = read_config('build_dedispts_from_fil.cfg')

# Set default values for empty dictionary items.
# Bandpass information stored on disk
if (dict['bandpass_file']==''):
    dict['bandpass_file'] = dict['basename']+'_bandpass.npz'
if (dict['BANDPASS_DIR']==''):
    dict['BANDPASS_DIR'] = dict['DATA_DIR']
# Rfimask
if (dict['mask_file']==''):
    dict['mask_file'] = dict['basename']+'_rfimask.npz'
if (dict['MASK_DIR']==''):
    dict['MASK_DIR'] = dict['DATA_DIR']
# Remove zerodm?
if (dict['do_zerodm']==''):
    dict['do_zerodm'] = False
# DM for dedispersion
if (dict['DM']==''):
    dict['DM'] = 0.0
# No. of blocks in which data are to be extracted
if (dict['N_blocks']==''):
    dict['N_blocks'] = 1
# Subbands
if (dict['use_subbands']==''):
    dict['use_subbands'] = False
# Polarization of interest
if (dict['pol']==''):
    dict['pol'] = 0
# Smoothing parameters
if (dict['do_smoothing']==''):
    dict['do_smoothing'] = True
if (dict['smoothing_method']==''):
    dict['smoothing_method'] = 'Blockavg2D'
if (dict['convolution_method']==''):
    dict['convolution_method'] = 'fftconvolve'
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time() # Start time
print('Processing file: ',dict['basename'])
hdr = Header(dict['DATA_DIR']+dict['fil_file'],file_type='filterbank') # Returns a Header object
tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
t_samp  = hdr.t_samp   # Sampling time (s)
chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
nchans  = hdr.nchans   # No. of channels
npol    = hdr.npol     # No. of polarizations
n_bytes = hdr.primary[b'nbits']/8.0 # No. of bytes per data sample
hdr_size = hdr.primary['hdr_size'] # Header size (bytes)

# Set up frequency array. Frequencies in GHz.
freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3
# Flip frequency axis if chan_bw<0.
if (chan_bw<0):
    freqs_GHz = np.flip(freqs_GHz)
    print('Frequencies arranged in ascending order.')

# Open the stored bandpass information from disk.
contents = np.load(dict['BANDPASS_DIR']+dict['bandpass_file'],allow_pickle=True)
median_bp = contents['Median bandpass']

# Open rfimask.
rfimask_contents = np.load(dict['MASK_DIR']+dict['mask_file'],allow_pickle=True)
mask_zap_chans_per_int = rfimask_contents['Channel mask per int']
ptsperint = rfimask_contents['Nptsperint']

# Discard data beyond specified band edges.
if dict['use_subbands']:
    freqs_extract = contents['Band edges (GHz)']
    ind_band_low = np.where(freqs_GHz>=np.max([dict['freq_band_low'],np.min(freqs_extract)]))[0][0]
    ind_band_high = np.where(freqs_GHz<=np.min([dict['freq_band_high'],np.max(freqs_extract)]))[0][-1]+1
else:
    ind_band_low = np.where(freqs_GHz>=dict['freq_band_low'])[0][0]
    ind_band_high = np.where(freqs_GHz<=dict['freq_band_high'])[0][-1]+1
freqs_GHz = freqs_GHz[ind_band_low:ind_band_high]
median_bp = median_bp[ind_band_low:ind_band_high]

# Block average the frequency axis and identify channels to be flagged in block-averaged data.
print('Identifying frequencies to be flagged post smoothing...')
if dict['do_smoothing']=='Blockavg2D':
    freqs_ds = blockavg1d(freqs_GHz,dict['kernel_size_freq_chans'])
else:
    freqs_ds = freqs_GHz
freqs_good = np.array([]) # List to store set of frequencies covered in each subband after smoothing.
for j in range(len(freqs_extract)):
    ind_low = np.where(freqs_ds>=freqs_extract[j,0])[0][0]
    ind_high = np.where(freqs_ds<=freqs_extract[j,1])[0][-1]
    freqs_good = np.append(freqs_good,freqs_ds[ind_low:ind_high+1])
freqs_good = np.sort(freqs_good)
freqs_bad_indices = np.array([index for index in range(len(freqs_ds)) if freqs_ds[index] not in freqs_good ])

# Reference frequency and low frequency limit for dedispersion.
ref_freq = freqs_GHz[np.where(freqs_GHz<=np.max(freqs_extract))[0][-1]] # Top of the highest subband
freq_low = freqs_GHz[np.where(freqs_GHz>=np.min(freqs_extract))[0][0]] # Bottom of the lowest subband
# Determine length of dedispersed time series for given supplied DM.
max_tDM = calc_tDM(freq_low,dict['DM'],ref_freq)
max_tDM_samples = np.round(max_tDM/t_samp).astype(int)
tot_dedisp_tsamples = tot_time_samples - max_tDM_samples
print('Total no. of time samples in raw data = %d'% (tot_time_samples))
print('For DM = %.2f pc/cc, no. of time samples in dedispersed time series = %d'% (dict['DM'],tot_dedisp_tsamples))

# Intialize values of t_start and t_stop in sample numbers based on calculated block length.
block_length = calc_block_length(tot_time_samples,dict['N_blocks'],max_tDM_samples)
t_start_array = np.arange(dict['N_blocks'])*(block_length-max_tDM_samples)
t_stop_array = t_start_array+block_length
t_stop_array[-1] = tot_time_samples

# Open filterbank file.
f = open(dict['DATA_DIR']+dict['fil_file'],'rb')
# Final arrays to be stored to disk.
dedisp_time_array = np.array([]) # Times covered in dedispersed time series
dedisp_ts = np.array([]) # Dedispersed time series.
blkavg_dedisp_ds = np.array([]).reshape((len(freqs_ds),0)) # Block-averaged dedispersed dynamic spectrum.
# Output file name
file_name = dict['basename']+'_pulseprof_DM'+'%0.1f'% (dict['DM'])

for i in range(dict['N_blocks']):
    current_cursor_position = f.tell()
    t_start = t_start_array[i]
    t_stop = t_stop_array[i]
    times = np.arange(t_start,t_stop)*t_samp # Array of times (s) for this block.
    # Extract the raw dynamic spectrum.
    print('Extracting dynamic spectrum between times %.2f - %.2f s'% (times[0],times[-1]))
    data = load_fil_data(dict['fil_file'],dict['DATA_DIR'],t_start,t_stop,npol,nchans,n_bytes,f,hdr_size,dict['pol'],current_cursor_position)
    # Flip frequency axis of DS if channel bandwidth is negative.
    if (chan_bw<0):
        print('Flipping frequency axis of DS')
        data = np.flip(data,axis=0)
    # Discard bandpass edges.
    data = data[ind_band_low:ind_band_high,:]
    # Correct for bandpass shape.
    data = correct_bandpass(data,median_bp)
    if dict['do_zerodm']:
        # Remove additive temporal noise from data.
        data = remove_additive_time_noise(data)[0]
    if dict['do_smoothing']:
        # Smooth the data using a 2D filter (block average/Gaussian2D/Hanning/Hamming)
        data, freqs_ds, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],freqs_GHz,times)
    # Update time resolution based on smoothing performed/avoided.
    t_samp_smoothed = times[1]-times[0]
    # Mask data at "bad" frequencies with NaNs.
    data[freqs_bad_indices] = np.NaN

    # Dedisperse the whole dynamic spectrum at specified DM.
    start_time = times[0]
    print('Dedispersing dynamic spectrum at DM = %.2f pc/cc'% (dict['DM']))
    data, dedisp_times,dedisp_wh_timeseries = dedisperse_ds(data,freqs_ds,dict['DM'],ref_freq,freq_low,t_samp_smoothed,start_time)
    print('Dedispersion complete.')

    # Update arrays for storing info to disk.
    dedisp_ts = np.append(dedisp_ts,dedisp_wh_timeseries)
    dedisp_time_array = np.append(dedisp_time_array,dedisp_times)
    blkavg_dedisp_ds = np.concatenate((blkavg_dedisp_ds,data),1)

    # Save data products to NumPy file
    save_array = [dedisp_time_array, dedisp_ts, blkavg_dedisp_ds, freqs_ds]
    save_keywords = ['Time (s)','Dedisp time series','Dedisp DS','Frequencies (GHz)']
    np.save(dict['OUTPUT_DIR']+file_name+'.npy',save_array)

# Close the file cursor.
f.close()
# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
