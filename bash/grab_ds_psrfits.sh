#!/bin/bash

# Define environment variables.
BASENAME=puppi_58506_M87_1347_0001_0036_tstart1450.0_tstop1550.0
EXECDIR=/home/ella1/asuresh/psrpipe/exec
CFGDIR=/home/ella1/asuresh/psrpipe/config
LOGDIR=/home/ella1/asuresh/psrpipe/Log

# Create LOGDIR if non-existent.
mkdir -p $LOGDIR

# Run bandpass computation script.
python $EXECDIR/grab_ds_psrfits.py -i $CFGDIR/grab_ds_psrfits.cfg | tee $LOGDIR/grab_ds_psrfits_$BASENAME.log
