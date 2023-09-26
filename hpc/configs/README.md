# README

This directory contains files containing hdyra overrides for array job batching for hpc.

Each file contains one list of overrides per line, as the array batch job will use the array job ID together with awk to index a line and sed the overrides as commandline args to our main script. 

This main script is currently for training automata, because this can take several ours but differ only by a few parameters, and so it is well suited for hpc array jobs.

To submit, use `sbatch array-job.sub`. To check the result of the submission, use `squeue -u panteater`.

For more info on array jobs on HPC, see https://rcic.uci.edu/slurm/examples.html.
