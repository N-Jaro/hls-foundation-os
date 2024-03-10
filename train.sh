#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=a100
#SBATCH --account=bbym-hydro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --output=/projects/bbym/nathanj/hls-foundation-os/map_seg_1.out

#activate conda
conda activate geo_foundation

# Load required modules
module load anaconda3_gpu cudnn

# Change directory
cd /projects/bbym/nathanj/hls-foundation-os

mim train mmsegmentation configs/map_seg.py --launcher pytorch --gpus 2
