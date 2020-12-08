#!/bin/bash

# Define environment variables.
BASENAME=M87_58547_0320_0026_0049
EXECDIR=/home/ella1/asuresh/psrpipe/exec
CFGDIR=/home/ella1/asuresh/psrpipe/config
LOGDIR=/home/ella1/asuresh/psrpipe/Log

# Create LOGDIR if non-existent.
mkdir -p $LOGDIR

# Run bandpass computation script.
python $EXECDIR/bandpass.py -i $CFGDIR/bandpass.cfg | tee $LOGDIR/bandpass_$BASENAME.log
