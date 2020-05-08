# Analyze (plot, FFT, folding) dedispersed timeseries produced using build_dedispts_from_fil.py or build_dedispts_from_psrfits.py and break into subbands if specified.

from psrdynspec import read_config
from psrdynspec.modules.filters1d import savgol_lowpass, blockavg1d
from psrdynspec.modules.filters2d import blockavg_ds
from psrdynspec.modules.normalize_Nd import normalize_stdnormal
from psrdynspec.modules.fft import fft1d_mask
from psrdynspec.modules.folding_plan import ProcessingPlan
from psrdynspec.modules.fold import fold_ts, fold_metric_periods, execute_plan
from psrdynspec.plotting.config import *
from psrdynspec.plotting.fold_plot import subplots_metric_profile

import os, time
import numpy as np
from astropy.stats import sigma_clip
############################################################################
# INPUTS
dict = read_config('analyze_customts.cfg')

# Plotting
if (dict['show_plot']==''):
    dict['show_plot'] = False
# Chop time.
if (dict['do_chop_time']==''):
    dict['do_chop_time'] = False
if (dict['start_time']==''):
    dict['start_time'] = 0.0
if (dict['stop_time']==''):
    dict['stop_time'] = -1
# Split into subbands
if (dict['do_split_subbands']==''):
    dict['do_split_subbands'] = False
if (dict['subband_edges']==''):
    dict['subband_edges'] = None
else:
    dict['subband_edges'] = np.array(dict['subband_edges'])
if (dict['combine_subbands']==''):
    dict['combine_subbands'] = None
    N_combinations = 0
else:
    dict['combine_subbands'] = np.array(dict['combine_subbands'])
    N_combinations = len(dict['combine_subbands'])
# Block averaging.
if (dict['do_blkavg']==''):
    dict['do_blkavg'] = False
if (dict['blkavg_time_factor']==''):
    dict['blkavg_time_factor'] = 1
if isinstance(dict['blkavg_time_factor'],float):
    dict['blkavg_time_factor'] = np.round(dict['blkavg_time_factor']).astype(int)
# Detrending
if (dict['do_detrend']==''):
    dict['do_detrend'] = False
if (dict['window_length_time']==''):
    dict['window_length_time'] = 1.0
if (dict['poly_degree']==''):
    dict['poly_degree'] = 1
# Timeseries plot
if (dict['plot_timeseries']==''):
    dict['plot_timeseries'] = False
# FFT
if (dict['do_FFT']==''):
    dict['do_FFT'] = False
if (dict['Nfft']==''):
    dict['Nfft'] = None
if (dict['search_scale']==''):
    dict['search_scale'] = 'log'
if (dict['exclude_noise']==''):
    dict['exclude_noise'] = False
if (dict['N_sigma']==''):
    dict['N_sigma'] = 3.0
if (dict['niter']==''):
    dict['niter'] = 5
# Time-domain folding
if (dict['do_time_domain_folding']==''):
    dict['do_time_domain_folding'] = False
if (dict['bins_min']==''):
    dict['bins_min'] = 128
if (dict['folding_metric']==''):
    dict['folding_metric'] = 'reducedchisquare'
if (dict['fold_index']==''):
    dict['fold_index'] = 0
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Function defintions

# Detrend an array of timeseries.
'''
Inputs:
timeseries = 2D array of shape (No. of timeseries, length of each timeseries)
start_freqs = 1D array, start frequency (GHz) of each timeseries
stop_freqs = 1D array, stop frequency (GHz) of each timeseries
'''
def detrend_timeseries_sets(timeseries, window_length_time, poly_degree, t_resol, start_freqs, stop_freqs):
    window_length = int(2*((window_length_time/t_resol)//2)+1) # Window length (odd number of samples)
    detrended_timeseries = []
    for j in range(len(timeseries)):
        print('Detrending timeseries from frequencies %.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]))
        trend = savgol_lowpass(timeseries[j],window_length, poly_degree)
        detrended_timeseries.append(timeseries[j] - trend)
    detrended_timeseries = np.array(detrended_timeseries)
    return detrended_timeseries

# Normalize sets of timeseries such that the peak of the frequency-integrated timeseries is set to unity.
'''
Inputs:
timeseries = 2D array of shape (No. of timeseries, length of each timeseries)
index_integrated = Index corresponding to the timeseries obtained by integrating over all frequencies.
'''
def normalize_unitypeak(timeseries,index_integrated):
    normalization_factor = 1./np.nanmax(timeseries[index_integrated])
    renormalized_timeseries = timeseries*normalization_factor
    return renormalized_timeseries
############################################################################
# Profile code execution.
prog_start_time = time.time()

# Load data.
print('Loading contents of %s' %(dict['npy_file']))
times, ts, ds, freqs = np.load(dict['NPY_DIR']+dict['npy_file'],allow_pickle=True)
t_resol = times[1] - times[0] # Time resolution (s)
print('Data loaded.')

# Chop timeseries if specified.
if dict['do_chop_time']:
    if (dict['stop_time'] == -1):
        dict['stop_time'] = times[-1]
    print('Selecting times:  %.2f <= t <= %.2f s'% (dict['start_time'],dict['stop_time']))
    select_time_indices = np.where(np.logical_and(times>=dict['start_time'],times<=dict['stop_time']))[0]
    ds = ds[:,select_time_indices]
    ts = ts[select_time_indices]
    times = times[select_time_indices]

# Block average data.
if dict['do_blkavg']:
    print('Block averaging data along time by factor %d'% (dict['blkavg_time_factor']))
    ds, freqs, times = blockavg_ds(ds, 1, dict['blkavg_time_factor'], freqs, times)
    ts = np.nansum(ds,axis=0)
    t_resol = times[1] - times[0] # Update time resolution after block averaging.

# Split data into subbands.
if dict['do_split_subbands']:
    N_subbands = len(dict['subband_edges'])
    start_freqs = np.array([freqs[0]]) # Record start frequencies of different timeseries to process.
    stop_freqs = np.array([freqs[-1]]) # Record stop frequencies of different timeseries to process.
    timeseries = np.array([ts]) # Different timeseries to process.
    for i in range(N_subbands):
        print('Extracting subband %d timeseries between frequencies %.2f - %.2f GHz'% (i,dict['subband_edges'][i,0], dict['subband_edges'][i,1]))
        start_index = np.where(freqs>=dict['subband_edges'][i,0])[0][0]
        stop_index = np.where(freqs<=dict['subband_edges'][i,1])[0][-1]
        start_freqs = np.append(start_freqs,freqs[start_index])
        stop_freqs = np.append(stop_freqs,freqs[stop_index])
        subband_ts = np.sum(ds[start_index:stop_index+1],axis=0) # Subband timeseries
        timeseries = np.append(timeseries,[subband_ts],axis=0)
    # Combine subbands if specified.
    if (dict['combine_subbands'] is not None):
        for j in range(N_combinations):
            print('Combining timeseries from subbands %s'% (dict['combine_subbands'][j]))
            min_freq = np.min(start_freqs[dict['combine_subbands'][j]+1])
            max_freq = np.max(stop_freqs[dict['combine_subbands'][j]+1])
            start_freqs = np.append(start_freqs,min_freq)
            stop_freqs = np.append(stop_freqs,max_freq)
            combined_ts = np.sum(timeseries[dict['combine_subbands'][j]+1],axis=0)
            timeseries = np.append(timeseries,[combined_ts],axis=0)
    N_timeseries = len(timeseries)
else:
    N_timeseries = 1
    N_subbands = 1
    start_freqs = np.array([freqs[0]])
    stop_freqs = np.array([freqs[-1]])
    timeseries = np.array([ts])

# Detrend timeseries.
if dict['do_detrend']:
    timeseries = detrend_timeseries_sets(timeseries, dict['window_length_time'], dict['poly_degree'], t_resol, start_freqs, stop_freqs)

# Normalize the peak of frequency-integrated timeseries to unity.
timeseries = normalize_unitypeak(timeseries,0)

# Plot timeseries.
if dict['plot_timeseries']:
    if dict['do_split_subbands']:
        # Plot summed timeseries and subband timeseries.
        print('Plotting subband timeseries')
        plot_name = dict['basename']+'_timeseries_subband.png'
        if dict['combine_subbands'] is not None:
            fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(6,8),sharex=True,gridspec_kw={'height_ratios': [1/(N_combinations+5), N_combinations/(N_combinations+5), 4/(N_combinations+5)]})
            # Plot timeseries obtained by combining specific groups of subbands.
            for j in range(N_subbands+1, N_subbands+N_combinations+1):
                axes[1].plot(times, (j-N_subbands-1)*1.5+timeseries[j],label = '%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]),color='k',linewidth=0.5)
                axes[1].annotate('%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]), xycoords='data',xy=(0.75*np.max(times),(j-N_subbands-1)*1.5+1.1),fontsize=12)
            axes[1].set_ylim((np.min(timeseries[N_subbands+1]), N_combinations*1.5))
        else:
            fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(6,8),sharex=True,gridspec_kw={'height_ratios': [1/5, 4/5]})
        # Plot band-integrated timeseries.
        axes[0].plot(times,timeseries[0],color='k',linewidth=0.5)
        axes[0].annotate('%.2f - %.2f GHz'% (start_freqs[0],stop_freqs[0]), xycoords='data',xy=(0.75*np.max(times),1.1),fontsize=12)
        axes[0].set_ylim((np.nanmin(timeseries[0]), 1.5))
        # Plot subband timeseries.
        for j in range(N_subbands,0,-1):
            axes[-1].plot(times,(N_subbands-j)+timeseries[j],label = '%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]),color='k',linewidth=0.5)
            axes[-1].annotate('%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]), xycoords='data',xy=(0.75*np.max(times),(N_subbands-j)+0.5),fontsize=12)
        axes[-1].set_xlabel('Time (s)',fontsize=14)
        axes[-1].set_ylim((np.min(timeseries[N_subbands]),N_subbands-0.2))
        fig.text(0.04, 0.5, 'Flux (normalized)', va='center', rotation='vertical', fontsize=14)
        fig.subplots_adjust(hspace=0)
        plt.savefig(dict['OUTPUT_DIR']+plot_name)
        if dict['show_plot']:
            plt.show()
        else:
            plt.close()
    else:
        # Plotting timeseries.
        print('Plotting timeseries')
        plot_name = dict['basename']+'_timeseries.png'
        plt.plot(times,timeseries[0],color='k')
        plt.annotate('%.2f - %.2f GHz'% (start_freqs[0],stop_freqs[0]), xycoords='axes fraction',xy=(0.75,0.9),fontsize=12)
        plt.ylabel('Flux (arb. units)',fontsize=14)
        plt.xlabel('Time (s)',fontsize=14)
        plt.ylim((np.min(timeseries[0]), np.max(timeseries[0])*1.3))
        plt.savefig(dict['OUTPUT_DIR']+plot_name)
        if dict['show_plot']:
            plt.show()
        else:
            plt.close()

if dict['do_FFT']:
    power_spectrum = []  # Power spectral density for different timeseries
    peak_fourier_freq = [] # Peak fourier frequency for each time eries
    for j in range(N_timeseries):
        if dict['Nfft'] is None:
            Nfft = len(timeseries[j])
        print('Computing FFT of %.2f - %.2f GHz timeseries'% (start_freqs[j],stop_freqs[j]))
        positive_freqs, powerspec_at_posfreqs, peak_indices, peak_freqs, peak_powerspec_values = fft1d_mask(timeseries[j], Nfft, t_resol, dict['exclude_noise'], dict['search_scale'], dict['niter'], dict['N_sigma'])
        power_spectrum.append(powerspec_at_posfreqs)
        if j==0:
            fourier_freqs = positive_freqs
    power_spectrum = normalize_unitypeak(np.array(power_spectrum),0)

    print('Plotting power spectra')
    plot_name = dict['basename']+'_powerspectra.png'
    log_fourier_freqs = np.log(fourier_freqs)
    annotation_lower_x_limit = np.exp(np.min(log_fourier_freqs) + 100*(np.max(log_fourier_freqs) - np.min(log_fourier_freqs))/150)
    if dict['do_split_subbands']:
        if dict['combine_subbands'] is not None:
            fig_fft, axes = plt.subplots(nrows=3,ncols=1,figsize=(6,8),sharex=True,gridspec_kw={'height_ratios': [1/(N_combinations+5), N_combinations/(N_combinations+5), 4/(N_combinations+5)]})
            # Plot power spectrum of timeseries obtained by combining specific groups of subbands.
            for j in range(N_subbands+1, N_subbands+N_combinations+1):
                axes[1].semilogx(fourier_freqs, (j-N_subbands-1)*1.5+power_spectrum[j],label = '%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]),color='k',linewidth=0.5)
                axes[1].annotate('%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]), xycoords='data',xy=(annotation_lower_x_limit,(j-N_subbands-1)*1.5+1.2),fontsize=12)

            axes[1].set_ylim((-0.2, N_combinations*1.5))
        else:
            fig_fft, axes = plt.subplots(nrows=2,ncols=1,figsize=(6,8),sharex=True,gridspec_kw={'height_ratios': [1/5, 4/5]})
        # Plot power spectrum of band-integrated timeseries.
        axes[0].semilogx(fourier_freqs,power_spectrum[0],color='k',linewidth=0.5)
        axes[0].annotate('%.2f - %.2f GHz'% (start_freqs[0],stop_freqs[0]), xycoords='data',xy=(annotation_lower_x_limit,1.2),fontsize=12)
        axes[0].set_ylim((-0.2, 1.5))

        # Plot power spectrum of  timeseries.
        for j in range(N_subbands,0,-1):
            axes[-1].semilogx(fourier_freqs,(N_subbands-j)+power_spectrum[j],label = '%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]),color='k',linewidth=0.5)
            axes[-1].annotate('%.2f - %.2f GHz'% (start_freqs[j],stop_freqs[j]), xycoords='data',xy=(annotation_lower_x_limit,(N_subbands-j)+0.5),fontsize=12)
        axes[-1].set_xlabel('Fourier frequency (Hz)',fontsize=14)
        axes[-1].set_ylim((-0.2,N_subbands-0.2))
        fig_fft.text(0.04, 0.5, 'Power spectral density (normalized)', va='center', rotation='vertical', fontsize=14)
        fig_fft.subplots_adjust(hspace=0)
        plt.savefig(dict['OUTPUT_DIR']+plot_name)
        if dict['show_plot']:
            plt.show()
        else:
            plt.close()
    else:
        fig_fft = plt.figure()
        plt.plot(fourier_freqs,power_spectrum[0],color='k')
        plt.annotate('%.2f - %.2f GHz'% (start_freqs[0],stop_freqs[0]), xycoords='axes fraction',xy=(0.75,0.9),fontsize=12)
        plt.ylabel('Power spectral density (arb. units)',fontsize=14)
        plt.xlabel('Fourier frequency (Hz)',fontsize=14)
        plt.xlim((0,100))
        plt.savefig(dict['OUTPUT_DIR']+plot_name)
        if dict['show_plot']:
            plt.show()
        else:
            plt.close()

if dict['do_time_domain_folding']:
    print('Folding frequency-summed timeseries at a large number of trial periods')
    fold_basename = dict['basename']+'_freqs%.2fto%.2f'% (start_freqs[dict['fold_index']], stop_freqs[dict['fold_index']])
    integrated_ts = normalize_stdnormal(timeseries[dict['fold_index']])
    nsamp = len(integrated_ts)
    plan = ProcessingPlan.create(nsamp, t_resol, dict['bins_min'], dict['P_min'], dict['P_max'])
    print(plan)
    print('Total no. of trial periods = %d'% (len(plan.periods)))
    metric_values, global_metricmax_index, global_metricmax, best_period, optimal_bins, optimal_dsfactor = execute_plan(integrated_ts, times, plan, dict['folding_metric'])
    profile, phasebins = fold_ts(blockavg1d(integrated_ts,optimal_dsfactor), blockavg1d(times,optimal_dsfactor), best_period, optimal_bins)
    subplots_metric_profile(plan.periods,metric_values,dict['folding_metric'],phasebins,profile,best_period,fold_basename,dict['OUTPUT_DIR'],dict['show_plot'])


# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
