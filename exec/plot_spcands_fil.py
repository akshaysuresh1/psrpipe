'''
This scripts peforms the following tasks.
1. Extract a list of single pulse candidates and their properties from PRESTO .singlepulse files.
2. Filter single pulse candidates based on their TOAs, DMs, and detection significance.
3. Grab and process (bandpass correction, zerodm removal, data smoothing) a chunk of filterbank data containing the candidate of interest.
4. Generate single pulse search output.
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from psrdynspec import read_config, Header, load_fil_data
from psrdynspec.io.read_rfifindmask import read_rfimask, modify_zapchans_bandpass
from psrdynspec.io.parse_sp import gen_singlepulse, remove_duplicates, apply_sigma_cutoff, remove_duplicates
from psrdynspec.io.rw_prestoinf import infodata
from psrdynspec.modules.dedisperse import calc_tDM, dedisperse_ds
from psrdynspec.modules.ds_systematics import remove_additive_time_noise, calc_median_bandpass, correct_bandpass
from psrdynspec.modules.filters1d import blockavg1d
from psrdynspec.modules.filters2d import smooth_master
from psrdynspec.plotting.config import *
from psrdynspec.plotting.spcands_plot import plot_DMtime, spcand_verification_plot
# Load standard packages.
from mpi4py import MPI
import numpy as np
import os, time, sys, glob
from argparse import ArgumentParser
#########################################################################
# Execute call for processors.
def myexecute(cand_index, cand_DMs, cand_sigma, cand_dedisp_times, metadata, int_times, mask_zap_chans, mask_zap_chans_per_int, freqs_GHz, tot_time_samples, t_samp, chan_bw, npol, nchans, n_bytes, hdr_size, hotpotato, f, rank):
    print('RANK %d: Working on candidate at index %d'% (rank, cand_index))
    DM = cand_DMs[cand_index] # DM (pc/cc) of single pulse candidate
    cand_time = cand_dedisp_times[cand_index] # Candidate time (s)
    t_ex = calc_tDM(freqs_GHz[0], DM, freqs_GHz[-1]) # Extra time of data to be loaded around the candidate time
    if DM<15.0:
        t_ex = np.max([0.2, t_ex])
    # Index of start time of data chunk to be loaded.
    if hotpotato['t_before'] is not None:
        tstart = np.max([0., np.floor((cand_time - hotpotato['t_before'])/t_samp) ]).astype(int)
    else:
        tstart = np.max([0, np.floor((cand_time - t_ex)/t_samp) ]).astype(int)
    # Index of stop time of data chunk to be loaded.
    if hotpotato['t_after'] is not None:
        tstop = np.min([tot_time_samples, np.floor((cand_time + hotpotato['t_after'])/t_samp) ]).astype(int)
    else:
        tstop = np.min([tot_time_samples, np.floor((cand_time + 2*t_ex)/t_samp) ]).astype(int)
     # 1D array of times (s)
    times = np.arange(tstart,tstop)*t_samp
    # Data chunk to load
    data = load_fil_data(f,tstart,tstop,npol,nchans,n_bytes,hdr_size,hotpotato['pol'],f.tell())

    # Flip frequency axis of DS if channel bandwidth is negative.
    if (chan_bw<0):
        print('RANK %d: Flipping frequency axis of DS'% (rank))
        data = np.flip(data,axis=0)
    # Clip bandpass edges.
    data = data[hotpotato['ind_band_low']:hotpotato['ind_band_high']]

    # Compute bandpass if needed.
    if hotpotato['bandpass_method']=='compute':
        hotpotato['median_bp'] = calc_median_bandpass(data)
    # Correct data for bandpass shape.
    print('RANK %d: Correcting data for bandpass shape'% (rank))
    if 0 in hotpotato['median_bp']:
        indices_zero_bp = np.where(hotpotato['median_bp']==0)[0]
        replace_value = np.median(hotpotato['median_bp'][np.where(hotpotato['median_bp']!=0)[0]])
        hotpotato['median_bp'][indices_zero_bp] = replace_value
        data[indices_zero_bp] = replace_value
    data = correct_bandpass(data, hotpotato['median_bp'])

    # Remove zerodm signal.
    if hotpotato['remove_zerodm']:
        data = remove_additive_time_noise(data)[0]
        print('RANK %d: Zerodm removal completed.'% (rank))

    # Apply rfifind mask on data.
    if hotpotato['apply_rfimask']:
        idx1 = np.where(int_times<=times[0])[0][-1]
        idx2 = np.where(int_times<times[-1])[0][-1] + 1
        cand_nint = idx2 - idx1
        cand_int_times = int_times[idx1:idx2]
        cand_mask_zap_chans_per_int = mask_zap_chans_per_int[idx1:idx2]
        # Boolean rfifind mask
        boolean_rfimask = np.zeros(data.shape,dtype=bool)
        for i in range(cand_nint):
            if i==cand_nint-1:
                tstop_int = tstop
            else:
                tstop_int = np.min(np.where(times>=cand_int_times[i+1])[0])
            tstart_int = np.min(np.where(times>=cand_int_times[i])[0])
            boolean_rfimask[cand_mask_zap_chans_per_int[i],tstart_int:tstop_int] = True
        print('RANK %d: Applying RFI mask on data'% (rank))
        data = np.ma.MaskedArray(data,mask=boolean_rfimask)
        # Replaced masked entries with mean value.
        print('RANK %d: Replacing masked entries with mean values'% (rank))
        data = np.ma.filled(data, fill_value=np.nanmean(data))

    # Smooth and/or downsample the data.
    kernel_size_time_samples = hotpotato['downsamp_time'][np.where(np.array(hotpotato['low_dm_cats'])<=DM)[0][-1]]
    data, freqs_GHz_smoothed, times = smooth_master(data,hotpotato['smoothing_method'],hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],kernel_size_time_samples,freqs_GHz,times)
    if hotpotato['smoothing_method']!='Blockavg2D':
        data, freqs_GHz_smoothed, times = smooth_master(data,'Blockavg2D',hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],kernel_size_time_samples,freqs_GHz_smoothed,times)

    # Remove residual spectral trend.
    data = data - np.median(data,axis=1)[:,None]

    # Dedisperse the data at DM of candidate detection.
    dedisp_ds, dedisp_times, dedisp_timeseries = dedisperse_ds(data, freqs_GHz_smoothed, DM, freqs_GHz_smoothed[-1], freqs_GHz_smoothed[0], times[1]-times[0], times[0])

    # Candidate verification plot
    mask_zap_check = list(np.sort(mask_zap_chans)//hotpotato['kernel_size_freq_chans'])
    mask_chans = np.array([chan for chan in np.unique(mask_zap_check) if mask_zap_check.count(chan)==hotpotato['kernel_size_freq_chans']])
    spcand_verification_plot(cand_index, cand_dedisp_times, cand_DMs, cand_sigma, metadata, data, times, freqs_GHz_smoothed, dedisp_ds, dedisp_timeseries, dedisp_times, SAVE_DIR=hotpotato['OUTPUT_DIR'], output_formats=hotpotato['output_formats'], show_plot=hotpotato['show_plot'], low_DM_cand=hotpotato['low_DM_cand'], high_DM_cand=hotpotato['high_DM_cand'], mask_chans=mask_chans, vmin=np.mean(data)-2*np.std(data), vmax=np.mean(data)+5*np.std(data), cmap=hotpotato['cmap'])

    # Write smoothed dynamic spectrum to disk as .npz file.
    if hotpotato['write_npz']:
        npz_filename = hotpotato['OUTPUT_DIR'] + '/' + hotpotato['basename'] + '_t%.2f_DM%.1f'% (cand_time, DM)
        write_npz_data(data, freqs_GHz_smoothed, times, mask_chans, npz_filename)

# Write dynamic spectrum and relevant metadata to disk as .npz file.
def write_npz_data(data, freqs_GHz, times, mask_chans, filename):
    save_array = [data, freqs_GHz, times, mask_chans]
    save_keywords = ['DS', 'Radio frequency (GHz)', 'Time (s)', 'Channel mask']
    np.savez(filename,**{name:value for name, value in zip(save_keywords, save_array)})

#  Filter single pulse candidates.
def filter_spcands(hotpotato):
    # Generate list of .singlepulse files.
    print('Generating list of .singlepulse files')
    singlepulse_filelist = sorted(glob.glob(hotpotato['SINGLEPULSE_DIR']+'/'+hotpotato['glob_singlepulse']))
    print('No. of .singlepulse files = %d'% (len(singlepulse_filelist)))
    # Metadata are read from the .inf file associated with the first .singlepulse file.
    print('Reading metadata')
    metadata = infodata(singlepulse_filelist[0].split('.singlepulse')[0]+'.inf')
    # Modify metadata as required.
    if hotpotato['source'] is not '':
        metadata.object = hotpotato['source']
    if hotpotato['instrument'] is not '':
        metadata.instrument = hotpotato['instrument']
    if hotpotato['basename'] is not '':
        metadata.basename = hotpotato['basename']
    center_freq = metadata.lofreq + 0.5*(metadata.numchan-1)*metadata.chan_width # MHz
    # Collate single pulse candidates in specified DM range from .singlepulse files.
    cand_DMs, cand_sigma, cand_dedisp_times, cand_dedisp_samples = gen_singlepulse(hotpotato['low_DM_cand'],hotpotato['high_DM_cand'],singlepulse_filelist)
    # Discard candidates that belong to time spans to be excluded.
    if len(hotpotato['exc_low_times'])>0:
        for i in range(len(hotpotato['exc_low_times'])):
            print('Excluding candidates between times %.2f - %.2f s'% (hotpotato['exc_low_times'][i], hotpotato['exc_high_times'][i]))
            select_indices = np.where(np.logical_and(cand_dedisp_times>=hotpotato['exc_low_times'][i], cand_dedisp_times<=hotpotato['exc_high_times'][i]))[0]
            # Delete selected index entries.
            cand_DMs = np.delete(cand_DMs, select_indices)
            cand_sigma = np.delete(cand_sigma, select_indices)
            cand_dedisp_times = np.delete(cand_dedisp_times, select_indices)
            cand_dedisp_samples = np.delete(cand_dedisp_samples, select_indices)
    # Apply S/N cutoff.
    print('Discarding candidates with S/N < %.2f'% (hotpotato['sigma_cutoff']))
    cand_DMs, cand_sigma, cand_dedisp_times, cand_dedisp_samples = apply_sigma_cutoff(cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples,hotpotato['sigma_cutoff'])
    # Remove duplicate candidates associated with the same single pulse event.
    print('Removing duplicate candidates using a time margin of %.2f ms and a DM margin of %.1f pc/cc'% (hotpotato['time_margin']*1e3, hotpotato['DM_margin']))
    cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples, select_indices = remove_duplicates(cand_DMs,cand_sigma,cand_dedisp_times,cand_dedisp_samples,hotpotato['time_margin'],hotpotato['DM_margin'])
    plot_DMtime(cand_dedisp_times, cand_DMs, cand_sigma, metadata, hotpotato['OUTPUT_DIR'], hotpotato['output_formats'], hotpotato['show_plot'], hotpotato['low_DM_cand'], hotpotato['high_DM_cand'], select_indices)
    print('Total number of unique candidates: %d'% (len(select_indices)))
    return metadata, cand_DMs, cand_sigma, cand_dedisp_times, cand_dedisp_samples, select_indices

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['output_formats']=='':
        hotpotato['output_formats'] = ['.png']
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['cmap']=='':
        hotpotato['cmap']='viridis'
    if hotpotato['show_plot']=='':
        hotpotato['show_plot'] = False
    if hotpotato['write_npz']=='':
        hotpotato['write_npz'] = False
    if hotpotato['t_before']=='':
        hotpotato['t_before'] = None
    if hotpotato['t_after']=='':
        hotpotato['t_after'] = None
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
    if hotpotato['low_dm_cats']=='':
        hotpotato['low_dm_cats'] = [0.]
    if hotpotato['downsamp_time']=='':
        hotpotato['downsamp_time'] = [1]
    return hotpotato
#########################################################################
# MAIN MPI function
def __MPI_MAIN__(parser):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    stat = MPI.Status()
    nproc = comm.Get_size()
    # Parent processor
    if rank==0:
        print('STARTING RANK 0')
        # Profile code execution.
        prog_start_time = time.time()

        parse_args = parser.parse_args()
        # Initialize parameter values
        inputs_cfg = parse_args.inputs_cfg

        # Construct list of calls to run from shell.
        hotpotato = set_defaults(read_config(inputs_cfg))

        # Create output directory if non-existent.
        if not os.path.isdir(hotpotato['OUTPUT_DIR']):
            os.makedirs(hotpotato['OUTPUT_DIR'])
        # Load information on single pulse candidates.
        metadata, cand_DMs, cand_sigma, cand_dedisp_times, cand_dedisp_samples, select_indices = filter_spcands(hotpotato)

        # Read header of filterbank file.
        hdr = Header(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'],file_type='filterbank') # Returns a Header object
        tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
        t_samp  = hdr.t_samp   # Sampling time (s)
        chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
        nchans  = hdr.nchans   # No. of channels
        npol    = hdr.npol     # No. of polarizations
        n_bytes = hdr.primary['nbits']/8.0 # No. of bytes per data sample
        hdr_size = hdr.primary['hdr_size'] # Header size (bytes)
        # Set up frequency array. Frequencies in GHz.
        freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3
        print(hdr)

        # Flip frequency axis if chan_bw<0.
        if (chan_bw<0):
            print('Channel bandwidth is negative.')
            freqs_GHz = np.flip(freqs_GHz)
            print('Frequencies rearranged in ascending order.')
        # Chop bandpass edges.
        hotpotato['ind_band_low'] = np.where(freqs_GHz>=hotpotato['freq_band_low'])[0][0]
        hotpotato['ind_band_high'] = np.where(freqs_GHz<=hotpotato['freq_band_high'])[0][-1]+1
        # Clip bandpass edges.
        freqs_GHz = freqs_GHz[hotpotato['ind_band_low']:hotpotato['ind_band_high']]

        # Load median bandpass, if pre-computed.
        if hotpotato['bandpass_method']=='file':
            print('Loading median bandpass from %s'% (hotpotato['bandpass_npz']))
            hotpotato['median_bp'] = np.load(hotpotato['BANDPASS_DIR']+'/'+hotpotato['bandpass_npz'],allow_pickle=True)['Median bandpass']
            hotpotato['median_bp'] = hotpotato['median_bp'][hotpotato['ind_band_low']:hotpotato['ind_band_high']]
            print('Median bandpass loaded.')
        elif hotpotato['bandpass_method'] not in ['file', 'compute']:
            print('Unrecognized bandpass computation method.')
            sys.exit(1)

        # Load rfifind mask.
        print('Reading rfifind mask: %s'% (hotpotato['rfimask']))
        nint, int_times, ptsperint, mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int = read_rfimask(hotpotato['RFIMASK_DIR']+'/'+hotpotato['rfimask'])
        mask_zap_chans, mask_zap_chans_per_int = modify_zapchans_bandpass(mask_zap_chans, mask_zap_chans_per_int, hotpotato['ind_band_low'], hotpotato['ind_band_high'])

        if nproc==1:
            f = open(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'], 'rb')
            for i in range(len(select_indices)):
                cand_index = select_indices[i]
                myexecute(cand_index, cand_DMs, cand_sigma, cand_dedisp_times, metadata, int_times, mask_zap_chans, mask_zap_chans_per_int, freqs_GHz, tot_time_samples, t_samp, chan_bw, npol, nchans, n_bytes, hdr_size, hotpotato, f, rank)
            f.close()
        else:
            # Distribute candidates evenly among child processors.
            indices_dist_list = np.array_split(select_indices,nproc-1)

            # Send data to child processors.
            for indx in range(1,nproc):
                comm.send((indices_dist_list[indx-1], cand_DMs, cand_sigma, cand_dedisp_times, metadata, int_times, mask_zap_chans, mask_zap_chans_per_int, freqs_GHz, tot_time_samples, t_samp, chan_bw, npol, nchans, n_bytes, hdr_size, hotpotato), dest=indx, tag=indx)
            comm.Barrier() # Wait for all child processors to receive sent call.
            # Receive Data from child processors after execution.
            comm.Barrier()

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        print('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        indx_vals, cand_DMs, cand_sigma, cand_dedisp_times, metadata, int_times, mask_zap_chans, mask_zap_chans_per_int, freqs_GHz, tot_time_samples, t_samp, chan_bw, npol, nchans, n_bytes, hdr_size, hotpotato = comm.recv(source=0, tag=rank)
        comm.Barrier()
        print('STARTING RANK: ',rank)
        f = open(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'], 'rb')
        for i in range(len(indx_vals)):
            cand_index = indx_vals[i]
            myexecute(cand_index, cand_DMs, cand_sigma, cand_dedisp_times, metadata, int_times, mask_zap_chans, mask_zap_chans_per_int, freqs_GHz, tot_time_samples, t_samp, chan_bw, npol, nchans, n_bytes, hdr_size, hotpotato, f, rank)
        f.close()
        print('FINISHING RANK: ',rank)
        comm.Barrier()
        # Send completed status back to parent processor.
#########################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py plot_spcands_fil.py [-h] -i INPUTS_CFG

Argmunents in parenthesis are required numbers for an MPI run.

This scripts peforms the following tasks.
1. Extract a list of single pulse candidates and their properties from PRESTO .singlepulse files.
2. Filter single pulse candidates based on their TOAs, DMs, and detection significance.
3. Grab and process (bandpass correction, zerodm removal, data smoothing) a chunk of filterbank data containing the candidate of interest.
4. Generate single pulse search output.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for verifying authenticity of single pulse candidates in filterbank dynamic spectra."""
    parser = ArgumentParser(description=".",usage=usage(),add_help=False)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    # Run MPI-parallelized prepsubband.
    __MPI_MAIN__(parser)
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
