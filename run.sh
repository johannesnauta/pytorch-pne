#!/bin/bash
script=$1
plot=$2
num_cores=4
echo "Using $num_cores core(s)..."
#
mpiexec --oversubscribe -n $num_cores python $script 
if [ -n "$plot" ]
then 
    python "$plot"
fi
echo "Finished."
