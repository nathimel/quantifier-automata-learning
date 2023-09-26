#!/bin/sh

# load conda and activate environment
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.8.5/etc/profile.d/conda.sh
conda activate qal

# run programs
echo
echo 'python src/setup.py "$@"'
python src/setup.py "$@"

echo
echo 'python src/main.py "$@"'
python src/main.py "$@"
