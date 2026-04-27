#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --partition=academic
#SBATCH --mem=64g
#SBATCH -o ../Output/Training/slurm/train%j.out
#SBATCH -e ../Output/Training/slurm/train%j.err
#SBATCH --gres=gpu:1    

module load python
source ../../../.venv/bin/activate
module load cuda

# Set data directories
TRAIN_DIR="../Data/Generated/train"
VAL_DIR="../Data/Generated/val"

# Training parameters
NUM_EPOCHS=100
BATCH_SIZE=32
LR=0.0001
SEQUENCE_LENGTH=5
LSTM_HIDDEN=512

# Run training
python Train.py \
    --train_data_dir $TRAIN_DIR \
    --val_data_dir $VAL_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    -v

echo "Training complete! Check outputs at:"
echo "  - Logs: ../Output/Training/logs/VISUAL/"
echo "  - Checkpoints: ../Output/Training/checkpoints/VISUAL/"