#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=train
#SBATCH --time=6:00:00
#SBATCH --partition=academic
#SBATCH --mem=16g
#SBATCH -o training_files/unsupervised/u_train%j.out
#SBATCH -e training_files/unsupervised/u_train%j.err
#SBATCH --gres=gpu:1    


module load python
source ../../../.venv/bin/activate
module load cuda
python Train.py -t u
