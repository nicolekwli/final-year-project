#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GIM.vision.main_vision --grayscale --download_dataset --save_dir vision_experiment --num_epochs 50

echo "Testing the Greedy InfoMax Model for image classification"
python -m GIM.vision.downstream_classification --grayscale --model_path ./logs/vision_experiment --model_num 49
