# Neural Network in C

This is a functional Neural Network built from scratch in C. 
Its primary goal is to predict the initial parameters of a parabolic trajectory. 
However, the modular design allows it to predict any number of continuous values. 
(Some variable and function names are in French.)

## Features
- Perceptron: Basic building block of the network
- Couche (Layer): Modular network layers
- Reseau (Network): Neural network structure
- Courbes (Curves): Tools for plotting or computing curves
- Initialization: Parameter initialization routines
- Fonctions (Functions): Various helper functions

## Installation
A practical example is shown on the file Perceptron_test.
## Usage
Compile the code with your C compiler:
bash
gcc Perceptron_test.c -o Perceptron.test -lm
./Perceptron_test
