#!/bin/sh

echo "Visualising the Greedy InfoMax Model on vision data (MNIST)"

# GIM
python -m GIM.vision.main_visualise --grayscale --model_path ./logs/vision_experiment --model_num 0

## GIM SUPERVISED
#python -m GIM.vision.main_visualise --grayscale --model_path ./logs/vision_experiment --model_num 29 --loss 1


#CPC
#python -m GIM.vision.main_visualise --grayscale --model_path ./logs/vision_experiment --model_num 15 --model_splits 1

# fully supervised
#python -m GIM.vision.main_visualise --grayscale --model_path ./logs/vision_experiment --model_num 0 --model_splits 1 --loss 1

