#!/bin/bash

# Define environment variables.
BASENAME=A00_58737_0024_tstart23.5_tstop23.9
EXECDIR=/home/ella1/asuresh/psrpipe/exec
CFGDIR=/home/ella1/asuresh/psrpipe/config
LOGDIR=/home/ella1/asuresh/psrpipe/Log

# Create LOGDIR if non-existent.
mkdir -p $LOGDIR

# Run bandpass computation script.
python $EXECDIR/grab_ds_fil.py -i $CFGDIR/grab_ds_fil.cfg | tee $LOGDIR/grab_ds_fil_$BASENAME.log
