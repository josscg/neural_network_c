#ifndef COUCHE_H
#define COUCHE_H

#include "Perceptron.h"

typedef struct Couche {
    int n_perceptrons;
    Perceptron* perceptrons;

    double (*fonction_activation)(double);
    double (*d_fonction_activation)(double);

    double *inputs;   // taille = n_perceptrons
    double *outputs;  // taille = n_perceptrons
} Couche;

Couche initialisation_couche(int n_perceptrons, int n_inputs, double (*f_activation)(double), double (*d_f_activation)(double));
void free_couche(Couche *couche);

#endif // COUCHE_H

