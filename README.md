# Multi-Layer Perceptron (MLP) Neural Network from Scratch

This repository contains Python code for implementing a Multi-Layer Perceptron (MLP) neural network from scratch. MLP is a feedforward artificial neural network that consists of multiple layers, including an input layer, one or more hidden layers, and an output layer. This implementation consists of both MLP regressor and Classifier that are available as two separate objects

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Concepts](#concepts)
- [Requirements](#requirements)
- [Results](#results)
- [References](#references)

## Overview

In this project, you will find documented implementations for essential concepts, including:

- Basic Neural Networks: Understanding how data flows through the network to produce predictions.
- Activation Functions: The role of activation functions in introducing non-linearity to neural networks.
      -relU
      -Sigmoid
- Backpropagation: The mathematics and logic behind training neural networks using gradient descent.
- Loss Functions: Evaluating the performance of the network and guiding the optimization process.
      - Squared Error
- Customization: Alter the network's architecture, learning rates, and other hyperparameters to adapt it for different tasks.

## Features

- Solves basic Regression and Classification Problems.
- built-in functions to ensure that an End-to-End neural network is built using only the available object 
- Flexibility to experiment with different network architectures, learning rates, input data and analyze the corresponding results.
- Verbose functionality to view the progress while training.

## Concepts
### Neurons
Neurons, often referred to as artificial neurons or perceptrons, are the building blocks of neural networks. They are mathematical models inspired by biological neurons. A single neuron takes multiple input signals, applies weights to these inputs, sums them up, and then passes the result through an activation function to produce an output.

<img src="https://user-images.githubusercontent.com/94131187/265493139-adbb8783-4c63-480f-ac07-845f16624139.png" alt="image" width="620" height="420" />
Source: V7 Labs

### Back-Propagation
Backpropagation is a key algorithm for training neural networks. It is a form of supervised learning where the network learns by adjusting its weights and biases based on the error it makes in predicting the desired output. The process involves calculating the gradient of the loss function with respect to the network's parameters (weights and biases) and updating these parameters in the opposite direction of the gradient to minimize the loss.

<img src="https://user-images.githubusercontent.com/94131187/265493463-89f276b6-0bc0-42cd-be60-bb319433fa53.png" alt="image" width="400" height="220" />
Source: Medium

### Gradient Descent
Gradient descent is the optimization algorithm commonly used to update the parameters of neural networks during training. It aims to find the minimum of a loss function by iteratively adjusting the parameters in the direction of the steepest descent (negative gradient).

<img src="https://user-images.githubusercontent.com/94131187/265493709-94e3aa37-a9d4-4f1c-a8f7-43b5711557b9.png" alt="image" width="420" height="120" />
Source: Geeks For Geeks

## Requirements

- Environment: Jupyter Notebook (or any other Python IDE)
- Basic Libraries: numpy, pandas, matplotlib.etc

run the code with predefined functions and Examples or alter the code to satisfy unique requirements

## Results

- A Neural Network model built completely from scratch by using all the concepts and ideas behind deep learning models.
- An in-depth view of the functioning of a Neural Network.
- A deep learning model capable of solving basic Regression and Classification Problems.

## References

- ### Medium: How to Build a Neural Network From Scratch
  https://medium.com/swlh/how-to-build-a-neural-network-from-scratch-b712d59ae641
- ### Valerio Velardo - The Sound of AI: TRAINING A NEURAL NETWORK: Implementing backpropagation and gradient descent from scratch
  https://www.youtube.com/watch?v=Z97XGNUUx9o&t=2s&ab_channel=ValerioVelardo-TheSoundofAI
