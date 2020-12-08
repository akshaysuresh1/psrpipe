#!/bin/bash

# MPI settings
niceval=1
mpiproc=3

# Define environment variables.
BASENAME=puppi_58505_M87_1339_0001_0026
EXECDIR=/home/ella1/asuresh/psrpipe/exec
CFGDIR=/home/ella1/asuresh/psrpipe/config
LOGDIR=/home/ella1/asuresh/psrpipe/Log

# Create LOGDIR if non-existent.
mkdir -p $LOGDIR

# Run rfifind command.
nice -$niceval mpiexec -n $mpiproc python -m mpi4py $EXECDIR/plot_spcands_psrfits.py -i $CFGDIR/plot_spcands_psrfits.cfg | tee $LOGDIR/spcands_psrfits_$BASENAME.log
