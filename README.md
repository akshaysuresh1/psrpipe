# psrpipe
Pipeline scripts for processing pulsar and FRB dynamic spectra using modules defined in [psrdynspec](https://github.com/akshaysuresh1/psrdynspec).

## Dependencies
All scripts import [psrdynspec](https://github.com/akshaysuresh1/psrdynspec) and its dependencies.

## Execution
For every script basename, there exists a config file (.cfg extension under ```config``` folder) and an executable script (.py extension under ```exec``` folder). The .cfg file specifies the inputs that are imported by the .py file during run time.

Say that you wanted to calculate the bandpass shape from filterbank / psrfits data and save the bandpass information to disk.
1. Start by editing the inputs in ```config\calc_bandpass.cfg```.
2. Edit ```bash/bandpass.sh```.
3. From within the ```psrpipe``` repository, run ```. bash/bandpass.sh``` on the terminal.

Each .py script also reports code run times on the terminal at the end of execution.

It is recommended that you initiate a screen/tmux screen prior to bash script execution.

## Troubleshooting
Please submit an issue to voice any problems or requests.

Improvements to the code are always welcome.
