#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --account comsm0045
#SBATCH --mem 120GB
#SBATCH --gres gpu:2

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

bash setup_dependencies.sh
source activate infomax
conda install -c pytorch cudatoolkit=10.1
conda install -c pytorch pytorch=1.7.0+cu101
conda install -c pytorch torchaudio

# bash audio_traineval_GIM.sh
echo "Training the Greedy InfoMax Model on audio data (librispeech)"
~/.conda/envs/infomax/bin/python -u -m GIM.audio.main_audio --subsample --num_epochs 1000 --learning_rate 2e-4 --start_epoch 0 -i ../scratch/datasets/ -o ../scratch/ --save_dir audio_experiment

# echo "Testing the Greedy InfoMax Model for phone classification"
# ~/.conda/envs/infomax/bin/python -m GIM.audio.linear_classifiers.logistic_regression_phones --model_path ./logs/audio_experiment --model_num 999 -i /mnt/storage/home/nl17247/scratch/datasets/ -o .

# echo "Testing the Greedy InfoMax Model for speaker classification"
# ~/.conda/envs/infomax/bin/python -m GIM.audio.linear_classifiers.logistic_regression_speaker --model_path ./logs/audio_experiment --model_num 999 -i /mnt/storage/home/nl17247/scratch/datasets/ -o .
