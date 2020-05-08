# Search for periodicity in a PRESTO-produced dedispersed time series.

from psrdynspec import read_config
from psrdynspec.io.read_presto_dat import load_presto_dat
from psrdynspec.modules.filters1d import blockavg1d, savgol_lowpass
from psrdynspec.modules.fft import fft1d_mask
from psrdynspec.modules.fold import fold_ts, execute_plan, fold_rotations_ts
from psrdynspec.modules.folding_plan import ProcessingPlan
from psrdynspec.modules.normalize_Nd import normalize_stdnormal
from psrdynspec.plotting.config import *
from psrdynspec.plotting.fft_plot import fft_gridplot
from psrdynspec.plotting.fold_plot import subplots_metric_profile, plot_foldedprofile_rotations

from astropy.stats import sigma_clip
import os, time
import numpy as np
############################################################################
# INPUTS
dict = read_config('search_presto_dat.cfg')

# Data
if (dict['DM']==''):
    dict['DM'] = 0.0
if (dict['low_freq_data']==''):
    dict['low_freq_data'] = None
if (dict['high_freq_data']==''):
    dict['high_freq_data'] = None
# Plotting
if (dict['show_plot']==''):
    dict['show_plot'] = False
if (dict['plot_format']==''):
    dict['plot_format'] = '.png'
# Chop time.
if (dict['do_chop_time']==''):
    dict['do_chop_time'] = False
if (dict['start_time']==''):
    dict['start_time'] = 0.0
if (dict['stop_time']==''):
    dict['stop_time'] = -1
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
# Normalization
if (dict['do_normalize']==''):
    dict['do_normalize'] = False
# Time series clipping
if (dict['do_clipping']==''):
    dict['do_clipping'] = False
if (dict['sigmaclip']==''):
    dict['sigmaclip'] = 3.0
# FFT
if (dict['do_FFT']==''):
    dict['do_FFT'] = False
if (dict['Nfft']==''):
    dict['Nfft'] = None
if (dict['outliers_only']==''):
    dict['outliers_only'] = False
if (dict['N_sigma']==''):
    dict['N_sigma'] = 3.0
# FFT plotting
if (dict['max_fourierfreq_plot']==''):
    dict['max_fourierfreq_plot'] = 100.0
if (dict['special_fourierfreq']==''):
    dict['special_fourierfreq'] = None
# Time-domain folding
if (dict['do_timeseries_fold']==''):
    dict['do_timeseries_fold'] = False
if (dict['bins_min']==''):
    dict['bins_min'] = 128
if (dict['metric']==''):
    dict['metric'] = 'reducedchisquare'
if (dict['do_fold_rotations']==''):
    dict['do_fold_rotations'] = False
############################################################################
# Plotting labels
fourierfreq_units = 'Hz'
powerspec_units = 'arbitary units'
# Construct frequency annotation.
if (dict['low_freq_data'] is not None) and (dict['high_freq_data'] is not None):
    radiofreq_annotation = '%.2f $-$ %.2f GHz'% (dict['low_freq_data'], dict['high_freq_data'])
elif (dict['low_freq_data'] is not None) and (dict['high_freq_data'] is None):
    radiofreq_annotation = '%.2f GHz'% (dict['low_freq_data'])
elif (dict['low_freq_data'] is None) and (dict['high_freq_data'] is not None):
    radiofreq_annotation = '%.2f GHz'% (dict['high_freq_data'])
else:
    freq_annotation = ''
# Timeseries unit
if dict['do_normalize']:
    timeseries_unit = 'normalized'
else:
    timeseries_unit = 'a.u.'
powerspec_unit = 'arbitrary units'   # Power spectrum units
############################################################################
# Initialize array of special fourier frequencies.
if dict['special_fourierfreq'] is None:
    special_fourierfreq = np.array([])
elif (type(dict['special_fourierfreq'])==float or type(dict['special_fourierfreq'])==int):
    special_fourierfreq = np.array([dict['special_fourierfreq']])
elif type(dict['special_fourierfreq'])==list:
    special_fourierfreq = np.array(dict['special_fourierfreq'])   # Convert list to array.
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['OUTPUT_DIR']):
    os.makedirs(dict['OUTPUT_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time()

# Read dedispersed time series from input .dat file.
print('Loading contents of %s' %(dict['dat_file']))
t_resol= dict['t_samp'] # Keep track of time resolution of time series at different steps.
dedisp_ts = load_presto_dat(dict['DAT_DIR']+dict['dat_file']) # Dedispersed time series
times = np.arange(len(dedisp_ts))*t_resol # 1D array of times (s)
print('Data loaded.')

# Chop time axis.
if dict['do_chop_time']:
    if (dict['stop_time'] == -1):
        dict['stop_time'] = times[-1]
    print('Chopping out times %.2f - %.2f s'% (dict['start_time'],dict['stop_time']))
    select_time_indices = np.where(np.logical_and(times>=dict['start_time'],times<=dict['stop_time']))[0]
    dedisp_ts = dedisp_ts[select_time_indices]
    times = times[select_time_indices]

# Block average data.
if dict['do_blkavg']:
    print('Block averaging data...')
    dedisp_ts = blockavg1d(dedisp_ts,dict['blkavg_time_factor'])
    times = blockavg1d(times,dict['blkavg_time_factor'])
    t_resol = times[1] - times[0]
    print('Block averaging complete.')

# Detrend time series.
if dict['do_detrend']:
    print('Detrending time series')
    window_length = int(2*((dict['window_length_time']/t_resol)//2)+1) # Window length (odd number of samples)
    trend = savgol_lowpass(dedisp_ts,window_length,dict['poly_degree'])
    dedisp_ts = dedisp_ts - trend

# Normalize the time series to zero median and unit variance.
if dict['do_normalize']:
    print('Normalizing time series to zero median and unit variance')
    dedisp_ts = normalize_stdnormal(dedisp_ts)

# Time series clipping.
if dict['do_clipping']:
    clip_array = sigma_clip(dedisp_ts, sigma=dict['sigmaclip'], cenfunc='median', stdfunc='std')
    median = np.ma.median(clip_array)
    std = np.std(clip_array)
    threshold = median+dict['sigmaclip']*std
    print('Clipping time series values less than %.2f'% (threshold))
    dedisp_ts = np.ma.masked_less(dedisp_ts, threshold)

# FFT of time series.
if dict['do_FFT']:
    if dict['Nfft'] is None:
        dict['Nfft'] = len(dedisp_ts)
    frequencies, power_spectrum, peak_indices, peak_freqs, peak_powerspec = fft1d_mask(np.ma.filled(dedisp_ts,0.0), dict['Nfft'], t_resol, dict['outliers_only'], 'linear', 5, dict['N_sigma'])
    # Sort peak frequencies in decreasing order of their significance.
    peak_ordering = np.argsort(peak_powerspec)[::-1]
    peak_indices = peak_indices[peak_ordering]
    peak_freqs = peak_freqs[peak_ordering]
    peak_powerspec = peak_powerspec[peak_ordering]
    print('10 largest FFT peaks are at fourier frequencies (Hz):')
    print(np.round(peak_freqs[:10],4))
    fft_gridplot(times, dedisp_ts, frequencies, power_spectrum, dict['max_fourierfreq_plot'], timeseries_unit, powerspec_unit, dict['DM'], radiofreq_annotation, special_fourierfreq, dict['basename'], dict['OUTPUT_DIR'], dict['show_plot'], dict['plot_format'])

# Time-domain folding
if dict['do_timeseries_fold']:
    if np.ma.is_masked(dedisp_ts):
        fold_basename = dict['basename']+'_noisemasked'
        modified_dedisp_ts = np.ma.filled(dedisp_ts,0.0) + 0.05*np.random.randn(len(dedisp_ts))
    else:
        modified_dedisp_ts = dedisp_ts
        fold_basename = dict['basename']
    nsamp = len(modified_dedisp_ts)
    plan = ProcessingPlan.create(nsamp, t_resol, dict['bins_min'], dict['P_min'], dict['P_max'])
    print(plan)
    metric_values, global_metricmax_index, global_metricmax, best_period, optimal_bins, optimal_dsfactor = execute_plan(modified_dedisp_ts, times, plan, dict['metric'])
    integrated_ts = blockavg1d(modified_dedisp_ts,optimal_dsfactor)
    integrated_times = blockavg1d(times,optimal_dsfactor)
    # Fold time series at best period.
    profile, phasebins = fold_ts(integrated_ts, integrated_times, best_period, optimal_bins)
    subplots_metric_profile(plan.periods, metric_values, dict['metric'], phasebins, profile, best_period, fold_basename, dict['OUTPUT_DIR'], dict['show_plot'])
    # Plot folded timeseries per rotation period that maximizes the chosen metric.
    if dict['do_fold_rotations']:
        profile_rotations, counts_perrot_phibin, phibins = fold_rotations_ts(integrated_ts, integrated_times, best_period, optimal_bins)
        plot_foldedprofile_rotations(profile_rotations,counts_perrot_phibin,phibins,fold_basename,dict['OUTPUT_DIR'],dict['show_plot'],low_phase_limit=0.0,high_phase_limit=1.0,rot_spacing = 1.0, normalization = 'quarterrotmax')

# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE ! HURRAY!
############################################################################
