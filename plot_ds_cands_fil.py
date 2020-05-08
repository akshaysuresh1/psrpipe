# Plot dynamic spectra of single pulse candidates identified using PRESTO single_pulse_search.py in a single .fil file.

from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.io.parse_sp import gen_singlepulse, remove_duplicates, apply_sigma_cutoff
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, correct_bandpass
from psrdynspec.modules.dedisperse import dedisperse_ds, calc_DM_at_maxSNR
from psrdynspec.modules.filters1d import savgol_lowpass
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.modules.subbands import construct_ds_from_subbands
from psrdynspec.plotting.config import *
from psrdynspec.plotting.spcands_plot import plot_diag_cands
from psrdynspec.plotting.ds_plot import plot_ds
from psrdynspec.plotting.dedisperse_plot import plot_dedisp_ds_SNRvsDM, plot_dedisp_subband_SNRvsDM, plot_dedispersed_ds

import os, time
import numpy as np
import glob
############################################################################
# INPUTS
dict = read_config('plot_ds_cands_fil.cfg')

# Set default values for empty dictionary items.
# Show plot?
if (dict['show_plot']==''):
    dict['show_plot'] = False
if (dict['log_colorbar']==''):
    dict['log_colorbar'] = False
if (dict['vmin_percentile']==''):
    dict['vmin_percentile'] = None
if (dict['vmax_percentile']==''):
    dict['vmax_percentile'] = None
if (dict['zero_centraltime']==''):
    dict['zero_centraltime'] = False
# Single pulse candidate properties
if (dict['SINGLEPULSE_DIR']==''):
    dict['SINGLEPULSE_DIR'] = dict['SINGLEPULSE_DIR']
if (dict['low_DM_cand']==''):
    dict['low_DM_cand'] = 0.0
if (dict['high_DM_cand']==''):
    dict['high_DM_cand'] = 3000.0
if (dict['sigma_cutoff']==''):
    dict['sigma_cutoff'] = 8.0
if (dict['time_margin']==''):
    dict['time_margin'] = 1.0e-2
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
# Data chunk extraction
if (dict['t_before']==''):
    dict['t_before'] = 0.1
if (dict['t_after']==''):
    dict['t_after'] = 1.0
# Subband edges
if (dict['use_subbands']==''):
    dict['use_subbands'] = False
# Polarization
if (dict['pol']==''):
    dict['pol'] = 0
# Smoothing parameters
if (dict['smoothing_method']==''):
    dict['smoothing_method'] = 'Bloackavg2D'
if (dict['convolution_method']==''):
    dict['convolution_method'] = 'fftconvolve'
# Dedispersion
if (dict['do_dedisperse']==''):
    dict['do_dedisperse'] = False
if (dict['low_trialDM']==''):
    dict['low_trialDM'] = 0
if (dict['high_trialDM']==''):
    dict['high_trialDM'] = 500
if (dict['num_DMs']==''):
    dict['num_DMs'] = 1
# Save products in .npz?
if (dict['save_npz']==''):
    dict['save_npz'] = True
############################################################################
# Units for plots
time_unit = 's'
time_offset_unit = 'ms'
timeoffset_conversion_factor = 1.0e3 # Conversion factor from time offsets (above specified units) to times (s).
freq_unit = 'GHz'
flux_unit = 'arb. units'
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time()

# Extract candidates from .singlepulse files.
cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples = gen_singlepulse(dict['low_DM_cand'],dict['high_DM_cand'],dict['glob_singlepulse'],dict['SINGLEPULSE_DIR'])
cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples = apply_sigma_cutoff(cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples,dict['sigma_cutoff'])
unique_cand_DMs, unique_cand_sigma, unique_cand_dedisp_times, unique_cand_dedisp_samples = remove_duplicates(cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples,dict['time_margin'])
print('Total number of candidates:',len(unique_cand_DMs))
plot_diag_cands(cand_dedisp_times,cand_DMs,unique_cand_dedisp_times,unique_cand_DMs,cand_sigma,dict['low_DM_cand'],dict['high_DM_cand'],dict['time_margin'],dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'])

# Read header.
print('Reading header of file %s'% (dict['fil_file']))
hdr = Header(dict['DATA_DIR']+dict['fil_file'],file_type='filterbank')
tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
t_samp  = hdr.t_samp   # Sampling time (s)
chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
nchans  = hdr.nchans   # No. of channels
npol    = hdr.npol     # No. of polarizations
hdr_size = hdr.primary['hdr_size'] # Header size (bytes)
n_bytes = hdr.primary[b'nbits']/8.0 # No. of bytes per pixel of dynamic spectrum
freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3 # 1D array of radio frequencies (GHz)

# Flip the frequency axis if channel bandwidth is negative.
if (chan_bw<0):
    freqs_GHz = np.flip(freqs_GHz)
    print('Frequencies arranged in ascending order.')

# Open rfimask.
rfimask_contents = np.load(dict['MASK_DIR']+dict['mask_file'],allow_pickle=True)
mask_zap_chans_per_int = rfimask_contents['Channel mask per int']
ptsperint = rfimask_contents['Nptsperint']
timeperint = ptsperint*t_samp # Time corresponding to one integration.

# Open the stored bandpass information from disk.
contents = np.load(dict['BANDPASS_DIR']+dict['bandpass_file'],allow_pickle=True)
median_bp = contents['Median bandpass']
freqs_extract = contents['Band edges (GHz)']

# Discard data beyond specified band edges.
ind_band_low = np.where(freqs_GHz>=dict['freq_band_low'])[0][0]
ind_band_high = np.where(freqs_GHz<=dict['freq_band_high'])[0][-1]+1
freqs_GHz = freqs_GHz[ind_band_low:ind_band_high]
median_bp = median_bp[ind_band_low:ind_band_high]

# Open filterbank file.
f = open(dict['DATA_DIR']+dict['fil_file'],'rb')

for i in range(len(unique_cand_dedisp_times)):
    current_cursor_position = f.tell()
    t_center = unique_cand_dedisp_times[i]
    presto_DM = unique_cand_DMs[i]
    sigma = unique_cand_sigma[i]
    print('WORKING ON CANDIDATE %d at time %.2f s'% (i+1,t_center))

    # Intialize arrray of times.
    t_start = (t_center-dict['t_before'])//t_samp # Sample number of start time
    t_stop = (t_center+dict['t_after'])//t_samp # Sample number of stop time
    times = np.arange(t_start,t_stop)*t_samp # 1D array of times
    int_numbers = np.arange(t_start, t_stop)//ptsperint # 1D array of integration nos. for each time sample
    unique_int_numbers = np.unique(int_numbers).astype(int)

    # Load raw DS from filterbank file.
    print('Reading in raw dynamic spectrum...')
    raw_ds = load_fil_data(dict['fil_file'],dict['DATA_DIR'],t_start,t_stop,npol,nchans,n_bytes,f,hdr_size,dict['pol'],current_cursor_position)

    # Flip along the frequency axis if channel bandwidth is negative.
    if (chan_bw<0):
        raw_ds = np.flip(raw_ds,axis=0)
        print('Flipped frequency axis of dynamic spectrum.')
    raw_ds = raw_ds[ind_band_low:ind_band_high,:] # Discard band edges.

    if (dict['use_subbands']==False):
        # Correct data for bandpass shape.
        data = correct_bandpass(raw_ds,median_bp)
        # Remove additive temporal noise from data.
        data, add_noise =  remove_additive_time_noise(data)
        # Mask RFI-affected elements of the dynamic spectrum.
        print('Masking RFI-affected data elements')
        for integration in unique_int_numbers:
            integration = int(integration)
            select_integrations = np.where(int_numbers==integration)[0]
            flag_channels = mask_zap_chans_per_int[integration]
            select_channels = flag_channels[np.where(np.logical_and(flag_channels>=ind_band_low,flag_channels<ind_band_high))[0]] - ind_band_low
            for k in select_channels:
                data[k,select_integrations] = np.NaN
        data = np.nan_to_num(data,nan=np.nanmean(data))
        # Smooth the data using the specified method and filter sizes.
        whole_ds, freqs_ds, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],freqs_GHz,times)
    else:
        # Split the data into subbands and operate.
        subband_data = [] # List to store subband data information.
        freqs_subband = [] # List to store set of frequencies covered in each subband after smoothing.
        if (dict['smoothing_method']!='Blockavg2D'):
            # Section the raw DS into subbands as specified.
            print('Sectioning raw DS into specified subbands..')
            for j in range(len(freqs_extract)):
                print('Working with the subband: %s - %s GHz'% (freqs_extract[j,0],freqs_extract[j,1]))
                ind_low = np.where(freqs_GHz>=freqs_extract[j,0])[0][0]
                ind_high = np.where(freqs_GHz>freqs_extract[j,1])[0][0]
                f_subband = freqs_GHz[ind_low:ind_high]
                data = raw_ds[ind_low:ind_high,:]
                # Correct data for bandpass shape.
                data = correct_bandpass(data,median_bp[ind_low:ind_high])
                # Remove additive time noise from data.
                data, add_noise =  remove_additive_time_noise(data)
                # Smooth the data with a suitable filter.
                data, f_subband, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],f_subband,times)
                # Update lists to reflect extracted subband information.
                subband_data.append(data)
                freqs_subband.append(f_subband)
            # Covert lists to arrays.
            subband_data = np.array(subband_data)
            freqs_subband = np.array(freqs_subband)
            whole_ds, freqs_ds = construct_ds_from_subbands(raw_ds.shape,subband_data,freqs_subband,freqs_GHz)
        else:
            # Correct data for bandpass shape.
            data = correct_bandpass(raw_ds,median_bp)
            # Remove additive time noise from data.
            data, add_noise =  remove_additive_time_noise(data)
            # Block average the data.
            whole_ds, freqs_ds, times = smooth_master(data,dict['smoothing_method'],dict['convolution_method'],dict['kernel_size_freq_chans'],dict['kernel_size_time_samples'],freqs_GHz,times)
            # Section the block-averaged DS into subbands as specified.
            print('Sectioning block-averaged DS into specified subbands..')
            for j in range(len(freqs_extract)):
                print('Working with the subband: %s - %s GHz'% (freqs_extract[j,0],freqs_extract[j,1]))
                ind_low = np.where(freqs_ds>=freqs_extract[j,0])[0][0]
                ind_high = np.where(freqs_ds>freqs_extract[j,1])[0][0]
                freqs_subband.append(freqs_ds[ind_low:ind_high])
                subband_data.append(whole_ds[ind_low:ind_high])
            # Covert lists to arrays.
            subband_data = np.array(subband_data)
            freqs_subband = np.array(freqs_subband)
            whole_ds, freqs_ds = construct_ds_from_subbands(whole_ds.shape,subband_data,freqs_subband,freqs_ds)
    t_samp_smoothed = times[1]-times[0] # Reset time resolution based on smoothing performed.

    # Detrend the dynamic spectrum
    print('Detrending dynamic spectrum')
    timeseries = np.nansum(whole_ds,axis=0)
    window_length_time = (dict['t_before']+dict['t_after'])/10
    window_length_samples = int(2*((window_length_time/t_samp_smoothed)//2)+1) # Window length (odd number of samples)
    trend = savgol_lowpass(timeseries,window_length_samples,1)/len(whole_ds) # 1-degree polynomial fit over a moving window of 0.1 s duration

    whole_ds = whole_ds - trend

    # vmin
    if (dict['vmin_percentile']!=None):
        vmin = np.nanpercentile(whole_ds,dict['vmin_percentile'])
    else:
        vmin = None
    # vmax
    if (dict['vmax_percentile']!=None):
        vmax = np.nanpercentile(whole_ds,dict['vmax_percentile'])
    else:
        vmax = None
    # Set central time to zero if specified.
    if (dict['zero_centraltime']==True):
        times_plot = (times - t_center)*timeoffset_conversion_factor
        time_unit_plot = time_offset_unit
    else:
        times_plot = times
        time_unit_plot = time_unit
    plot_ds(whole_ds,times_plot[0],times_plot[-1],freqs_ds[0],freqs_ds[-1],time_offset_unit,freq_unit,flux_unit,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'],vmin=vmin,vmax=vmax,log_colorbar=dict['log_colorbar'])

    # Dedisperse the DS if specified.
    if (dict['do_dedisperse']==True):
        trial_DMs = np.linspace(dict['low_trialDM'],dict['high_trialDM'],dict['num_DMs']) # Trial DMs for dedispersion
        ref_freq = np.max(freqs_ds) # Reference frequency for dedispersion (highest frequency in data)
        freq_low = np.min(freqs_ds) # Bottom frequency of the data
        start_time = times[0] # Start time (s) of smoothed data.

        # Dedisperse the whole DS at above supplied trial DMs.
        print('Dedispersing the entire dynamic spectrum')
        SNR_wh, signal_array, offpulse_std_value = calc_DM_at_maxSNR(whole_ds,freqs_ds,trial_DMs,ref_freq,freq_low,t_samp_smoothed,start_time,t_center)
        print('Calculating optimal DM...')
        max_SNR_wh = np.max(SNR_wh)
        optimal_DM_ds = trial_DMs[np.argmax(SNR_wh)]
        print(' Optimal DM = %.2f pc/cc'% (optimal_DM_ds))
        print('Beginning dedispersion at optimal DM')
        dedisp_wh_ds, dedisp_times,dedisp_wh_timeseries = dedisperse_ds(whole_ds,freqs_ds,optimal_DM_ds,ref_freq,freq_low,t_samp_smoothed,start_time)
        print('Dedispersion complete.')
        # Plot DS and SNR vs. DM.
        plot_dedisp_ds_SNRvsDM(whole_ds,times,dedisp_wh_ds,dedisp_wh_timeseries,dedisp_times,freqs_ds,t_center,t_samp_smoothed,trial_DMs,SNR_wh,optimal_DM_ds,offpulse_std_value,flux_unit,freq_unit,time_offset_unit,timeoffset_conversion_factor,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'],vmin,vmax,dict['log_colorbar'])
        # Plot dedispersed DS at optimal DM.
        # vmin
        if (dict['vmin_percentile']!=None):
            vmin = np.nanpercentile(dedisp_wh_ds,dict['vmin_percentile'])
        else:
            vmin = None
        # vmax
        if (dict['vmax_percentile']!=None):
            vmax = np.nanpercentile(dedisp_wh_ds,dict['vmax_percentile'])
        else:
            vmax = None
        # Set central time to zero if specified.
        if (dict['zero_centraltime']==True):
            times_plot = (dedisp_times - dedisp_times[np.argmax(dedisp_wh_timeseries)])*timeoffset_conversion_factor # ms
            time_unit_plot = time_offset_unit
        else:
            times_plot = dedisp_times
            time_unit_plot = time_unit
        plot_dedispersed_ds(dedisp_wh_ds,dedisp_wh_timeseries,times_plot,freqs_ds,t_center,t_samp_smoothed,optimal_DM_ds,time_unit_plot,freq_unit,flux_unit,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'],vmin=vmin,vmax=vmax,log_colorbar=dict['log_colorbar'])
        # Dedispersing the data separately in subbands, if specified.
        if (dict['use_subbands']==True):
            optimal_DM_subbands = np.zeros(len(freqs_extract))
            for j in range(len(freqs_extract)):
                print('Dedispersing subband: %s - %s GHz'% (freqs_extract[j,0],freqs_extract[j,1]))
                print('Calculating the DM that maximizes the S/N of the pulse in the dedispersed time series..')
                SNR, signal_array, offpulse_std_value = calc_DM_at_maxSNR(subband_data[j],freqs_subband[j],trial_DMs,ref_freq,freq_low,t_samp_smoothed,start_time,t_center)
                print('Calculation complete.')
                max_SNR = np.max(SNR)
                optimal_DM = trial_DMs[np.argmax(SNR)]
                optimal_DM_subbands[j] = optimal_DM

                print('Beginning dedispersion at optimal DM')
                dedisp_ds, dedisp_times,dedisp_timeseries = dedisperse_ds(subband_data[j],freqs_subband[j],optimal_DM,ref_freq,freq_low,t_samp_smoothed,start_time)
                print('Dedispersion complete.')
                plot_dedisp_subband_SNRvsDM(dedisp_ds,dedisp_timeseries,dedisp_times,freqs_subband[j],t_center,t_samp_smoothed,trial_DMs,SNR,optimal_DM,offpulse_std_value,freq_unit,time_offset_unit,timeoffset_conversion_factor,dict['basename'],dict['OUTPUT_DIR'],dict['show_plot'])
        else:
            optimal_DM_subbands = np.ones(len(freqs_extract))*np.NaN


    else:
        optimal_DM_ds = np.NaN
        optimal_DM_subbands = np.ones(len(freqs_extract))*np.NaN

    # Save data products to NumPy file
    if dict['save_npz']:
        file_name = dict['basename']+'_t'+'%07.3f'% (t_center)
        save_array = [whole_ds, freqs_ds, times, freqs_extract, optimal_DM_ds, optimal_DM_subbands, trial_DMs, SNR_wh]
        save_keywords = ['DS', 'Freqs (GHz)', 'Time (s)', 'Subband edges (GHz)', 'Optimal DS DM', 'Optimal subband DM', 'Trial DMs (pc/cc)', 'SNRs at trial DMs']
        np.savez(dict['OUTPUT_DIR']+file_name,**{name:value for name, value in zip(save_keywords,save_array)})

# Close the file cursor.
f.close()

# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
