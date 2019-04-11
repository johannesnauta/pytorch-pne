#!/bin/bash
script=$1
plot=$2
num_cores=4
echo "Using $num_cores core(s)..."
#
mpiexec -n $num_cores python $script 
if [ "$plot" == "plot" ]
then 
    python plot.py
fi
echo "Finished."