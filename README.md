# psrpipe
Pipeline scripts for processing pulsar and FRB dynamic spectra using modules defined in [psrdynspec](https://github.com/akshaysuresh1/psrdynspec).

## Dependencies
All scripts import [psrdynspec](https://github.com/akshaysuresh1/psrdynspec) and its dependencies. 

## Execution
For every script basename, there exists a .cfg extension and a .py extension. The .cfg file specifies the inputs that are imported by the .py file during run time.

Say that you wanted to calculate the bandpass shape from filterbank / psrfits data and save the bandpass information to disk.
1. Start by editing the inputs in "calc_bandpass.cfg."
2. From within the ```psrpipe``` repository, run ```python calc_bandpass.py``` on the terminal. 

Each .py script also reports code run times on the terminal at the end of execution.
