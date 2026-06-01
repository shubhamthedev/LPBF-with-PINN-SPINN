#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --job-name=lpbf_spinns
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --output=/home/y0113734/heat/zlog/output_%j.log  # Standard output

# Run your singularity command
singularity exec --nv /home/y0113734/heat/pideep.sif python3 -u pinns.py