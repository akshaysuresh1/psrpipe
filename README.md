# psrpipe
Pipeline scripts for processing pulsar and FRB dynamic spectra using modules defined in [psrdynspec](https://github.com/akshaysuresh1/psrdynspec).

## Dependencies
All scripts import [psrdynspec](https://github.com/akshaysuresh1/psrdynspec) and its dependencies.


## Repository Structure
For every script basename, there exists a config file (.cfg extension under ```config``` folder) and an executable script (.py extension under ```exec``` folder). A .cfg file specifies inputs that are passed on to its corresponding .py script during execution

Each .py script also reports run times on the terminal at the end of execution. Owing to potentially long run times, I recommend users to execute programs within a screen/tmux session.

## Non-MPI executable scripts:
1. ```bandpass.py```: Compute median bandpass shape based on a chunk of PSRFITS or filterbank data.
2. ```grab_ds_fil.py```: Plot smoothed, RFI-masked dynamic spectrum of a chunk of filterbank data.
2. ```grab_ds_psrfits.py```: Plot smoothed, RFI-masked dynamic spectrum of a chunk of PSRFITS data.

Program run syntax:
```python <path to executable file> -i <path to config file of inputs>``` <br>
The ```-i``` flag specifies the input configuration file to be read by the executable file.

Example call:
```python exec/bandpass.py -i config/bandpass.cfg``` <br>

## MPI-enabled executable scripts:
1. ```plot_spcands_fil.py```: Plot dynamic spectra of single pulse candidates identified in filterbank data.  
2. ```plot_spcands_psrfits.py```: Plot dynamic spectra of single pulse candidates identified in PSRFITS data.

Program run syntax:
```mpirun -n <nproc> python <path to executable file> -i <path to config file of inputs>``` <br>
Default execution assumes operation on a single processor. If multiple processors are called, a parent-child MPI framework is invoked. Within this model, one processor is designated as a parent processor, whereas the remaining processors are classified as child processors. The parent distributes tasks evenly and collates outputs from the child processors.

Example call:
```mpirun -n 4 python exec/plot_spcands_fil.py -i config/plot_spcands_fil.cfg```

## Ongoing development
The ```DEV``` folder contains code under development. Use of these scripts for data processing is highly discouraged.

## Troubleshooting
Please submit an issue to voice any problems or requests. Improvements to the code are always welcome.
