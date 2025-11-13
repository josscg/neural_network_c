#include "Perceptron.h"
#include "Couche.h"
#include "Reseau.h"
#include "Courbe.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* On crée la fonction qui initialize les poids et bias des perceptrons */

Perceptron initialisation_perceptron(int n_inputs) {
    Perceptron perceptron;
    perceptron.n_inputs = n_inputs;
    perceptron.poids = (double*)malloc(n_inputs * sizeof(double));
    perceptron.nouveaus_poids = (double*)malloc(n_inputs * sizeof(double));

    if (!perceptron.poids || !perceptron.nouveaus_poids) {
        fprintf(stderr, "Allocation failed in initialisation_perceptron\n");
        exit(EXIT_FAILURE);
    }

    // symmetric initialization in [-1, 1)
    for (int i = 0; i < n_inputs; i++) {
        perceptron.poids[i] = drand48() * 2.0 - 1.0;
        perceptron.nouveaus_poids[i] = perceptron.poids[i];
    }

    perceptron.bias = drand48() * 2.0 - 1.0;
    perceptron.nouveau_bias = perceptron.bias;

    return perceptron;
}


void free_perceptron(Perceptron *p) {
    if (!p) return;
    free(p->poids);
    free(p->nouveaus_poids);
    p->poids = NULL;
    p->nouveaus_poids = NULL;
}


Couche initialisation_couche(int n_perceptrons, int n_inputs, double (*f_activation)(double), double (*d_f_activation)(double)) {
    Couche couche;
    couche.n_perceptrons = n_perceptrons;
    couche.perceptrons = (Perceptron*)malloc(n_perceptrons * sizeof(Perceptron));
    if (!couche.perceptrons) {
        fprintf(stderr, "Allocation failed in initialisation_couche\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n_perceptrons; i++) {
        couche.perceptrons[i] = initialisation_perceptron(n_inputs);
    }

    couche.inputs = (double *)malloc(n_perceptrons * sizeof(double));
    couche.outputs = (double *)malloc(n_perceptrons * sizeof(double));
    if (!couche.inputs || !couche.outputs) {
        fprintf(stderr, "Allocation failed for couche arrays\n");
        exit(EXIT_FAILURE);
    }

    couche.fonction_activation = f_activation;
    couche.d_fonction_activation = d_f_activation;

    // initialize outputs/inputs to zero
    for (int i = 0; i < n_perceptrons; i++) {
        couche.inputs[i] = 0.0;
        couche.outputs[i] = 0.0;
    }

    return couche;
}

void free_couche(Couche *couche) {
    if (!couche) return;
    for (int i = 0; i < couche->n_perceptrons; i++) {
        free_perceptron(&couche->perceptrons[i]);
    }
    free(couche->perceptrons);
    free(couche->inputs);
    free(couche->outputs);
    couche->perceptrons = NULL;
    couche->inputs = NULL;
    couche->outputs = NULL;
}




Reseau initialisation_reseau(int n_couches){
    Reseau reseau;
    reseau.n_couches = n_couches;
    reseau.couches = (Couche*)malloc(n_couches * sizeof(Couche));
    if (!reseau.couches) {
        fprintf(stderr, "Allocation failed in initialisation_reseau\n");
        exit(EXIT_FAILURE);
    }
    // caller must fill each reseau.couches[i] by calling initialisation_couche and assigning
    return reseau;
}

void free_reseau(Reseau *reseau) {
    if (!reseau) return;
    for (int i = 0; i < reseau->n_couches; i++) {
        free_couche(&reseau->couches[i]);
    }
    free(reseau->couches);
    reseau->couches = NULL;
}

Courbe initialisation_courbe(int n_points){
    Courbe courbe;
    courbe.n_points = n_points;
    courbe.points = (double*)malloc(n_points * sizeof(double));
    courbe.parametres_expectes = (double*)malloc(3 * sizeof(double));
    courbe.parametres_expectes_normalises = (double*)malloc(3 * sizeof(double));

    if (!courbe.points || !courbe.parametres_expectes || !courbe.parametres_expectes_normalises) {
        fprintf(stderr, "Allocation failed in initialisation_courbe\n");
        exit(EXIT_FAILURE);
    }

    const double g = 9.81;
    const double h_min = 5; const double h_max = 20;
    const double v_min = 4;  const double v_max = 20;

    double v0 = drand48()*(v_max - v_min) + v_min;
    double alpha_deg = drand48() * (90 - 1) + 1;
    double alpha = alpha_deg * (3.1415 / 180.0);
    double h = drand48() * (h_max - h_min) + h_min;

    courbe.parametres_expectes[0] = h;
    courbe.parametres_expectes[1] = v0;
    courbe.parametres_expectes[2] = alpha;

    courbe.parametres_expectes_normalises[0] = (courbe.parametres_expectes[0] - h_min)/(h_max - h_min);
    courbe.parametres_expectes_normalises[1] = (courbe.parametres_expectes[1] - v_min)/(v_max - v_min);
    // normalize angle between ~1deg to pi/2 radians used earlier
    double amin = 1.0 * (3.1415/180.0);
    double amax = 90.0 * (3.1415/180.0);
    courbe.parametres_expectes_normalises[2] = (courbe.parametres_expectes[2] - amin)/(amax - amin);

    for (int i = 0; i < n_points; i++){
        double t = (double)i;
        double hauteur = -0.5*g*pow(t,2)/(pow(v0,2)*pow(cos(alpha),2)) + t*tan(alpha) + h;
        courbe.points[i] = hauteur;
    }
    return courbe;
}

void free_courbe(Courbe *courbe){
    if (!courbe) return;
    free(courbe->points);
    free(courbe->parametres_expectes);
    free(courbe->parametres_expectes_normalises);
    courbe->points = NULL;
    courbe->parametres_expectes = NULL;
    courbe->parametres_expectes_normalises = NULL;
}
