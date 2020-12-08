# Run an FFA search on a large number of dedispersed time-series over a range of trail periods.

# psrdynspec imports
from psrdynspec import read_config
from psrdynspec.modules.fft import fft1d_blocks_avg, find_FFTpeaks
from psrdynspec.modules.fold import select_metric_function
from psrdynspec.plotting.fft_plot import fft_gridplot
from psrdynspec.plotting.config import *
from psrdynspec.modules.folding_plan import ProcessingPlan
# Riptide imports
from riptide import TimeSeries, ffa1, ffaprd
# Standard imports
from astropy.stats import sigma_clip
import csv
import numpy as np
import os, glob, time
######################################################################
# INPUTS
dict = read_config('ffasearch_presto_dat.cfg')

# Data
if (dict['low_freq_data']==''):
    dict['low_freq_data'] = None
if (dict['high_freq_data']==''):
    dict['high_freq_data'] = None
# Output paths
if dict['FFT_DIR']=='':
    dict['FFT_DIR'] = dict['DAT_DIR']
if dict['FOLD_DIR']=='':
    dict['FOLD_DIR'] = dict['DAT_DIR']
# Plotting
if dict['show_plot']=='':
    dict['show_plot'] = False
if dict['plot_format']=='':
    dict['plot_format'] = '.png'
# Detrending
if dict['rmed_width']=='':
    dict['rmed_width'] = 1.0
# FFT
if dict['do_FFT']=='':
    dict['do_FFT'] = False
if dict['Nfft']=='':
    dict['Nfft'] = None
if dict['max_fourierfreq_plot']=='':
    dict['max_fourierfreq_plot'] = 100.0
if dict['special_fourierfreq']=='':
    dict['special_fourierfreq'] = None
if dict['write_FFTpeaks']=='':
    dict['write_FFTpeaks'] = False
# FFA
if dict['do_FFA']=='':
    dict['do_FFA'] = False
if dict['bins_min']=='':
    dict['bins_min'] = 128
if dict['bins_max']=='':
    dict['bins_max'] = 256
if dict['P_min']=='':
    dict['P_min'] = 1.0
if dict['P_max']=='':
    dict['P_max'] = 10.0
if dict['metric']=='':
    dict['metric'] = 'reducedchisquare'
############################################################################
# Construct frequency annotation.
if (dict['low_freq_data'] is not None) and (dict['high_freq_data'] is not None):
    radiofreq_annotation = '%.2f $-$ %.2f GHz'% (dict['low_freq_data'], dict['high_freq_data'])
elif (dict['low_freq_data'] is not None) and (dict['high_freq_data'] is None):
    radiofreq_annotation = '%.2f GHz'% (dict['low_freq_data'])
elif (dict['low_freq_data'] is None) and (dict['high_freq_data'] is not None):
    radiofreq_annotation = '%.2f GHz'% (dict['high_freq_data'])
else:
    radiofreq_annotation = ''
############################################################################
# Initialize array of special fourier frequencies.
if (type(dict['special_fourierfreq'])==float or type(dict['special_fourierfreq'])==int):
    dict['special_fourierfreq'] = np.array([dict['special_fourierfreq']]) # Cast integer or float to array of size 1.
elif type(dict['special_fourierfreq'])==list:
    dict['special_fourierfreq'] = np.array(dict['special_fourierfreq'])   # Convert list to array.
############################################################################
# Create output directory if non-existent.
if not os.path.isdir(dict['FFT_DIR']):
    os.makedirs(dict['FFT_DIR'])
if not os.path.isdir(dict['FOLD_DIR']):
    os.makedirs(dict['FOLD_DIR'])
############################################################################
# Profile code execution.
prog_start_time = time.time()

print('Parsing .dat files using glob string: %s'% (dict['glob_basename']))
inf_list = sorted(glob.glob(dict['DAT_DIR']+dict['glob_basename']+'.inf'))
Nfiles = len(inf_list)
print('No. of .dat files to process = %d \n'% (Nfiles))

for i in range(Nfiles):
    basename = inf_list[i].split('.inf')[0].split(dict['DAT_DIR'])[-1]
    print('File: %s'% (basename))

    # Read in dedispersed time-series as a riptide TimeSeries object.
    timeseries = TimeSeries.from_presto_inf(inf_list[i])
    tsamp = timeseries.tsamp # Sampling time (s)
    nsamp = timeseries.nsamp # No. of samples (s)
    times = np.arange(nsamp)*tsamp # 1D array of times (s)
    print('Sampling time (s) = %.4f ms'% (tsamp*1e3))
    print('No. of samples = %d'% (nsamp))

    # Read DM value from file name.
    if 'DM' in inf_list[i]:
        DM = float(inf_list[i].split('DM')[1].split('.inf')[0])
        basename = basename.split('DM')[0]+'DM%06.1f'% (DM)
    else:
        DM = None

    # Detrend the time-series.
    timeseries = timeseries.deredden(width=dict['rmed_width'])

    # Normalize time-series to zero mean and unit variance.
    timeseries = timeseries.normalise()

    # FFT of time-series.
    if dict['do_FFT']:
        Nfft = dict['Nfft']
        if dict['Nfft'] is None:
            Nfft = nsamp
        power_spectrum, frequencies = fft1d_blocks_avg(timeseries.data, Nfft, tsamp, remove_DC_spike=False,return_positive_freqs_only=True)
        power_spectrum = power_spectrum/np.std(power_spectrum)
        peak_indices, peak_freqs, peak_powerspec = find_FFTpeaks(frequencies, power_spectrum, search_scale='log', niter=1, Nsigma=5.0)
        # Normalize power spectrum to unit variance.
        fft_gridplot(times, timeseries.data, frequencies, power_spectrum, dict['max_fourierfreq_plot'], 'normalized', 'normalized to unit variance', DM, radiofreq_annotation, dict['special_fourierfreq'], basename, dict['FFT_DIR'], dict['show_plot'], dict['plot_format'])
        if dict['write_FFTpeaks']:
            header = ['Frequency (Hz)', 'Power spectrum S/N', 'Indices']
            CSV_DIR = dict['FFT_DIR']+'FFTpeaks_csv/'
            if not os.path.isdir(CSV_DIR):
                os.makedirs(CSV_DIR)
            l = [peak_freqs, peak_powerspec, peak_indices]
            zipped_rows = zip(*l)
            # Write .csv file to disk.
            with open(CSV_DIR+basename+'_FFTpeaks.csv','w') as csv_file:
                print('Writing information of FFT peaks to a .csv file')
                writer = csv.writer(csv_file,delimiter=',')
                writer.writerow(header)
                for line in zipped_rows:
                	writer.writerow(line)

    # FFA of time-series
    if dict['do_FFA']:
        plan = ProcessingPlan.create(nsamp, tsamp, dict['bins_min'], dict['P_min'], dict['P_max'])# bins_max = dict['bins_max'])
        print(plan)
        dsfactors = plan.octaves.dsfactor.values
        bins_min = plan.octaves.bins_min.values
        bins_max = plan.octaves.bins_max.values
        ffa_tsamp = plan.octaves.tsamp.values
        metric_function = select_metric_function(dict['metric'])
        Nsteps = len(dsfactors)
        metric_values = []
        trial_periods = []
        # Begin FFA.
        for j in range(Nsteps):
            if dsfactors[j]>1:
                downsampled_ts = timeseries.downsample(factor=dsfactors[j])
            else:
                downsampled_ts = timeseries
            for b in range(bins_min[j], bins_max[j]):
                print('Step %d, bins %d'% (j+1,b))
                base_period =  int(dsfactors[j]*b) # Base period in sample numbers
                ffa_transform = ffa1(downsampled_ts.data, base_period) # Outputs 2D FFA transform
                ffa_periods = ffaprd(downsampled_ts.nsamp, base_period, tsamp)
                ffa_metric = []
                for k in range(len(ffa_transform)):
                    ffa_metric.append(metric_function(ffa_transform[k]))
                trial_periods = np.append(trial_periods, ffa_periods)
                metric_values = np.append(metric_values, np.array(ffa_metric))
    print('\n')

# Calculate total run time for the code.
prog_end_time = time.time()
run_time = (prog_end_time - prog_start_time)/60.0
print('Code run time = %.2f minutes'% (run_time))
## END OF CODE !
############################################################################
