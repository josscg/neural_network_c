#include "Couche.h"
#include "Reseau.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>


/* On définit ici les possibles fonction d'activation
des couches. Chaque couche peut avoir sa propre fonction d'activation et ses dérivées*/

double sigmoid(double x){
    return 1/(1 + exp(-x));
    }

double d_sigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
    }

double sinus(double x){
    return sin(x);
    }

double d_sinus(double x){
    return cos(x);
    }

double tanh_act(double x){
    return tanh(x); 
    }

double d_tanh(double x){
    double t = tanh(x); return 1 - t*t; 
    }


/* Ici On définit ici une fonction qui calcule les inputs d'une couche en fonction des outputs 
de la couche précente, et qui ensuite calcule les ouputs d'une couche en fonction de ses inputs */

void calcul_inputs_outputs(Couche *couche, const double *outs_couche_avant) {
    for (int i = 0; i < couche->n_perceptrons; i++) {
        double res = 0.0;
        Perceptron *p = &couche->perceptrons[i];
        for (int j = 0; j < p->n_inputs; j++) {
            res += p->poids[j] * outs_couche_avant[j];
        }
        couche->inputs[i] = res + p->bias;
        couche->outputs[i] = couche->fonction_activation(couche->inputs[i]);
    }
}



/* On crée la fonction qui se charge de réaliser la forward propagation du réseau, en d'autres mots,
qui se charge de trouve les outputs de chaque couche  pour pouvoir comparer les output finales
avec les valeurs attendues. 
Ici on considère que pour toutes les couches, leur fonctions d'activation sont deja définies et
les couches ont été saisie dans reseau.couches */

void forward(Reseau *reseau, const double *points){
    // first layer: inputs come from 'points'
    calcul_inputs_outputs(&reseau->couches[0], points);
    for (int i = 1; i < reseau->n_couches; i++){
        calcul_inputs_outputs(&reseau->couches[i], reseau->couches[i-1].outputs);
    }
}


/* MSE over n outputs */
double mse(const double *x, const double *y, int n) {
    double erreur = 0.0;
    for (int i = 0; i < n; i++){
        double diff = x[i] - y[i];
        erreur += diff * diff;
    }
    return erreur / (double)n;
}

/* derivative of MSE wrt x (single element) if MSE = (1/n) sum (x-y)^2 */
double d_mse(double x, double y) {
    // Note: caller must know n; if used for n=3 use 2*(x-y)/3
    // To keep general, user of this function must divide by n externally if needed.
    return 2.0 * (x - y);
}


/* On crée la fonction qui se charge de réaliser la back propagation du réseau. 
Ici on considère que pour toutes les couches, leur fonctions d'activation et leur dérivés sont deja définies et
les couches ont été saisie dans reseau.couches*/
void backpropagation(Reseau *reseau, const double *inputs, const double *outputs_expectes, double learning_rate) {
    int L = reseau->n_couches;
    // allocate an array of delta arrays, one per layer
    double **deltas = (double**)malloc(L * sizeof(double*));
    if (!deltas) {
        fprintf(stderr, "Allocation failed for deltas\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < L; i++) deltas[i] = NULL;

    // iterate from last layer to first
    for (int c = L - 1; c >= 0; c--) {
        Couche *couche = &reseau->couches[c];
        deltas[c] = (double*)malloc(couche->n_perceptrons * sizeof(double));
        if (!deltas[c]) {
            fprintf(stderr, "Allocation failed for deltas[%d]\n", c);
            exit(EXIT_FAILURE);
        }

        if (c == L - 1) {
            // output layer: delta = dLoss/dout * d_out/d_in
            for (int i = 0; i < couche->n_perceptrons; i++) {
                double out = couche->outputs[i];
                double in = couche->inputs[i];
                double derreur_dout = d_mse(out, outputs_expectes[i]) / (double)couche->n_perceptrons; // average per-output
                deltas[c][i] = derreur_dout * couche->d_fonction_activation(in);

                // update nouveaus_poids using outputs of previous layer
                for (int w = 0; w < couche->perceptrons[i].n_inputs; w++) {
                    double prev_out = (c == 0) ? inputs[w] : reseau->couches[c-1].outputs[w];
                    couche->perceptrons[i].nouveaus_poids[w] = couche->perceptrons[i].poids[w] - learning_rate * deltas[c][i] * prev_out;
                }
                couche->perceptrons[i].nouveau_bias = couche->perceptrons[i].bias - learning_rate * deltas[c][i];
            }
        } else {
            // hidden or first hidden
            for (int i = 0; i < couche->n_perceptrons; i++) {
                double somme_delta = 0.0;
                // use deltas from layer c+1
                Couche *next = &reseau->couches[c+1];
                for (int j = 0; j < next->n_perceptrons; j++) {
                    somme_delta += deltas[c+1][j] * next->perceptrons[j].poids[i];
                }
                double in = couche->inputs[i];
                deltas[c][i] = somme_delta * couche->d_fonction_activation(in);

                for (int w = 0; w < couche->perceptrons[i].n_inputs; w++) {
                    double prev_out = (c == 0) ? inputs[w] : reseau->couches[c-1].outputs[w];
                    couche->perceptrons[i].nouveaus_poids[w] = couche->perceptrons[i].poids[w] - learning_rate * deltas[c][i] * prev_out;
                }
                couche->perceptrons[i].nouveau_bias = couche->perceptrons[i].bias - learning_rate * deltas[c][i];
            }
        }
    }

    // free deltas
    for (int i = 0; i < L; i++) {
        free(deltas[i]);
        deltas[i] = NULL;
    }
    free(deltas);
}




/* Ici on définit la fonction qui actualise les poids du réseau après avoir fini 
la backpropagation */
void actualiser_poids_bias(Reseau *reseau){
    int n_couches = reseau->n_couches;
    for (int c = 0; c < n_couches; c++) {
        Couche *couche = &reseau->couches[c];
        for (int p = 0; p < couche->n_perceptrons; p++) {
            Perceptron *perceptron = &couche->perceptrons[p];

            // Update weights and clamp any NaN/inf
            for (int w = 0; w < perceptron->n_inputs; w++) {
                perceptron->poids[w] = perceptron->nouveaus_poids[w];

                if (isnan(perceptron->poids[w]) || isinf(perceptron->poids[w])) {
                    perceptron->poids[w] = 0.1 * (drand48() - 0.5);
                }
            }

            // Update bias and clamp NaN/inf
            perceptron->bias = perceptron->nouveau_bias;
            if (isnan(perceptron->bias) || isinf(perceptron->bias)) {
                perceptron->bias = 0.0;
            }
        }
    }
}




/* On définit finalement la fonction qui entraine le réseau.
On crée ensuite la fonction qui prédit les paramètres d'une ou plusieurs courbes une foi entriné le réseau.  */

void entrainer(Reseau *reseau, const double *inputs, const double *outputs_expectes, double learning_rate){
    forward(reseau, inputs);
    backpropagation(reseau, inputs, outputs_expectes, learning_rate);
    actualiser_poids_bias(reseau);
}

double* predict(Reseau *reseau, const double *inputs){
    double* predictions = (double*)malloc(3 * sizeof(double));
    if (!predictions) {
        fprintf(stderr, "Allocation failed in predict\n");
        exit(EXIT_FAILURE);
    }
    forward(reseau, inputs);
    Couche *last = &reseau->couches[reseau->n_couches - 1];

    // safe check: require at least 3 outputs
    if (last->n_perceptrons < 3) {
        fprintf(stderr, "predict: last layer must have >= 3 outputs\n");
        exit(EXIT_FAILURE);
    }

    // Denormalize according to your previous scheme
    // h: [5,20], v: [4,20], alpha: [~0.01745, ~1.57075]
    predictions[0] = last->outputs[0] * (20.0 - 5.0) + 5.0;
    predictions[1] = last->outputs[1] * (20.0 - 4.0) + 4.0;
    predictions[2] = last->outputs[2] * (1.57075 - 0.01745278) + 0.01745278;

    return predictions;
}



void plot_erreur(const double *erreurs, int size) {
    FILE *data_file = fopen("erreurs.dat", "w");
    if (data_file == NULL) {
        fprintf(stderr, "Erreur ouvrant fichier erreurs.dat.\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(data_file, "%d %f\n", i, erreurs[i]);
    }
    fclose(data_file);

    FILE *gnuplot_script = fopen("plot_script.gnuplot", "w");
    if (gnuplot_script == NULL) {
        fprintf(stderr, "Erreur ouvrant fichier plot_script.gnuplot.\n");
        return;
    }
    fprintf(gnuplot_script, "set terminal pngcairo size 800,600\n");
    fprintf(gnuplot_script, "set output 'plot.png'\n");
    fprintf(gnuplot_script, "set xlabel 'Iterations'\n");
    fprintf(gnuplot_script, "set ylabel 'Erreur'\n");
    fprintf(gnuplot_script, "plot 'erreurs.dat' with linespoints title ''\n");
    fclose(gnuplot_script);

    system("gnuplot plot_script.gnuplot");
}

