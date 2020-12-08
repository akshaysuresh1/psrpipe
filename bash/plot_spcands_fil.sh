#!/bin/bash

# MPI settings
niceval=0
mpiproc=1

# Define environment variables.
BASENAME=A00_58737_0024
EXECDIR=/home/ella1/asuresh/psrpipe/exec
CFGDIR=/home/ella1/asuresh/psrpipe/config
LOGDIR=/home/ella1/asuresh/psrpipe/Log

# Create LOGDIR if non-existent.
mkdir -p $LOGDIR

# Run rfifind command.
nice -$niceval mpiexec -n $mpiproc python -m mpi4py $EXECDIR/plot_spcands_fil.py -i $CFGDIR/plot_spcands_fil.cfg | tee $LOGDIR/spcands_fil_$BASENAME.log
