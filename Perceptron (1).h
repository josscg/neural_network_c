#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct Perceptron {
    int n_inputs;
    double *poids;
    double bias;

    double *nouveaus_poids;
    double nouveau_bias;
} Perceptron;

Perceptron initialisation_perceptron(int n_inputs);
void free_perceptron(Perceptron *p);

#endif // PERCEPTRON_H
