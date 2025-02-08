# Drowsiness Detection Project

This folder contains the implementation of a drowsiness detection project developed by **Ali Sadeghifar**.

## Project Overview
The goal of this project is to develop a real-time system for detecting drowsiness in drivers by analyzing facial landmarks and eye movements. The system processes images to detect faces, extracts relevant features, and classifies the driver as "sleepy" or "not sleepy." The project leverages machine learning and computer vision techniques to achieve accurate detection.

## Project Structure
The project is divided into the following components:

- **dataset**: This directory contains the dataset of labeled images for training and testing the model.
- **model**: This directory includes the core implementation for training a CNN and MLP hybrid model, which analyzes both facial features and facial images.
- **results**: This directory contains saved models, training logs, and visualizations of the training process, such as accuracy and loss plots.

## Requirements
- TensorFlow
- Keras
- OpenCV
- dlib
- scikit-learn
- Matplotlib

## How to Use
1. Clone the repository.
2. Download the dataset and place it in the `dataset` folder.
3. Train the model by running the `train_model.py` script.
4. Use the trained model for predictions in the `predict.py` script.

## License
This project is licensed under the MIT License.
