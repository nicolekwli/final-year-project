#!/bin/sh

echo "Training the Greedy InfoMax Model on audio data (librispeech)"
python -m GIM.audio.main_audio --subsample --num_epochs 1000 --learning_rate 2e-4 --start_epoch 0 -i ./datasets/ -o . --save_dir audio_experiment

echo "Testing the Greedy InfoMax Model for phone classification"
python -m GIM.audio.linear_classifiers.logistic_regression_phones --model_path ./logs/audio_experiment --model_num 999 -i ./datasets/ -o .

echo "Testing the Greedy InfoMax Model for speaker classification"
python -m GIM.audio.linear_classifiers.logistic_regression_speaker --model_path ./logs/audio_experiment --model_num 999 -i ./datasets/ -o .
