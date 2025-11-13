#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "Perceptron.h"
#include "Couche.h"
#include "Reseau.h"
#include "Courbe.h"
#include "Initialisation.c"
#include "Fonctions.c"


/* Note: do NOT #include .c files. Compile them and link together. */

int main() {
    srand48(time(NULL));

    int num_courbes = 100;
    int num_points = 50;

    /* allocate array of Courbe (training set) */
    Courbe *training_set = (Courbe*)malloc(num_courbes * sizeof(Courbe));
    if (!training_set) {
        fprintf(stderr, "Allocation failed for training_set\n");
        return 1;
    }

    for (int i = 0; i < num_courbes; i++) {
        training_set[i] = initialisation_courbe(num_points);
    }

    /* initialize network with 8 layers and then initialize each layer */
    Reseau reseau = initialisation_reseau(8);

    reseau.couches[0] = initialisation_couche(60, num_points, sinus, d_sinus);
    reseau.couches[1] = initialisation_couche(50, reseau.couches[0].n_perceptrons, sinus, d_sinus);
    reseau.couches[2] = initialisation_couche(50, reseau.couches[1].n_perceptrons, sinus, d_sinus);
    reseau.couches[3] = initialisation_couche(30, reseau.couches[2].n_perceptrons, tanh_act, d_tanh);
    reseau.couches[4] = initialisation_couche(30, reseau.couches[3].n_perceptrons, tanh_act, d_tanh);
    reseau.couches[5] = initialisation_couche(10, reseau.couches[4].n_perceptrons, tanh_act, d_tanh);
    reseau.couches[6] = initialisation_couche(10, reseau.couches[5].n_perceptrons, tanh_act, d_tanh);
    reseau.couches[7] = initialisation_couche(3,  reseau.couches[6].n_perceptrons, sigmoid, d_sigmoid);

    int epochs = 50;
    double learning_rate = 0.01;
    double *erreurs_epochs = (double*)malloc(epochs * sizeof(double));
    if (!erreurs_epochs) {
        fprintf(stderr, "Allocation failed for erreurs_epochs\n");
        free_reseau(&reseau);
        for (int i = 0; i < num_courbes; i++) free_courbe(&training_set[i]);
        free(training_set);
        return 1;
    }

    for (int e = 0; e < epochs; e++) {
        double *erreurs_itertions = (double*)malloc(num_courbes * sizeof(double));
        if (!erreurs_itertions) {
            fprintf(stderr, "Allocation failed for erreurs_itertions\n");
            break;
        }

        for (int it = 0; it < num_courbes; it++) {
            Courbe *courbe = &training_set[it];

            /* entrainer expects pointer to reseau and normalized expected outputs */
            entrainer(&reseau, courbe->points, courbe->parametres_expectes_normalises, learning_rate);

            for (int i = 0; i < reseau.couches[reseau.n_couches - 1].n_perceptrons; i++) {
                if (isnan(reseau.couches[reseau.n_couches - 1].outputs[i])) {
                    printf("NaN detected after training sample %d, epoch %d, output %d\n", it, e, i);
                    exit(1);
                    }           
                }   

            /* compute mse between network outputs (last layer) and expected normalized params
               mse signature: mse(const double *x, const double *y, int n) */
            erreurs_itertions[it] = mse(reseau.couches[reseau.n_couches - 1].outputs,
                                        courbe->parametres_expectes_normalises,
                                        3);
        }

        double erreur_moyen = 0.0;
        for (int i = 0; i < num_courbes; i++) {
            erreur_moyen += erreurs_itertions[i];
        }
        erreur_moyen /= (double)num_courbes;
        erreurs_epochs[e] = erreur_moyen;

        free(erreurs_itertions);
    }

    /* plot error curve */
    plot_erreur(erreurs_epochs, epochs);

    /* predict on a fresh random curve */
    Courbe courbe_prediction = initialisation_courbe(num_points);
    double *predictions = predict(&reseau, courbe_prediction.points);

    printf("\nTrue (h, v0, alpha): %lf %lf %lf\n",
           courbe_prediction.parametres_expectes[0],
           courbe_prediction.parametres_expectes[1],
           courbe_prediction.parametres_expectes[2]);

    printf("Predicted (h, v0, alpha): %lf %lf %lf\n",
           predictions[0], predictions[1], predictions[2]);

    /* cleanup */
    for (int i = 0; i < num_courbes; i++) {
        free_courbe(&training_set[i]);
    }
    free(training_set);
    free(predictions);
    free_courbe(&courbe_prediction);
    free(erreurs_epochs);
    free_reseau(&reseau);

    return 0;
}



/*Reseau reseau = initialisation_reseau(5);
    reseau.couches[0] = initialisation_couche(50, num_points);
    reseau.couches[0].fonction_activation = sinus;
    reseau.couches[0].d_fonction_activation = d_sinus;

    reseau.couches[1] = initialisation_couche(30, reseau.couches[0].n_perceptrons);
    reseau.couches[1].fonction_activation = sinus;
    reseau.couches[1].d_fonction_activation = d_sinus;

    reseau.couches[2] = initialisation_couche(15, reseau.couches[1].n_perceptrons);
    reseau.couches[2].fonction_activation = sinus;
    reseau.couches[2].d_fonction_activation = d_sinus;

    reseau.couches[3] = initialisation_couche(7, reseau.couches[2].n_perceptrons);
    reseau.couches[3].fonction_activation = sinus;
    reseau.couches[3].d_fonction_activation = d_sinus;

    reseau.couches[4] = initialisation_couche(3, reseau.couches[3].n_perceptrons);
    reseau.couches[4].fonction_activation = sinus;
    reseau.couches[4].d_fonction_activation = d_sinus;*/


    /*forward(reseau, courbe.points);

    for (int i=0; i<reseau.n_couches; i++){
        Couche couche = reseau.couches[i];
        printf("\nInputs couche %i:\n", i);
        for (int j=0; j<couche.n_perceptrons; j++){
            printf("%lf ", couche.inputs[j]);
        }
        printf("\nOutputs couche %i:\n", i);
        for (int j=0; j<couche.n_perceptrons; j++){
            printf("%lf ", couche.outputs[j]);
        }
    }


    printf("\nOutput expectes\n");
    for (int i = 0; i<3; i++){
        printf("%lf ", courbe.parametres_expectes[i]);
    }
    printf("\n");

    
    backpropagation(reseau, courbe.points, courbe.parametres_expectes, learning_rate);*/




    /*
    Perceptron perceptron = reseau.couches[1].perceptrons[2];
    printf("poids du peceptron_2,3\n");
    for (int i=0; i<num_points; i++){
        printf("%lf\n", perceptron.poids[i]);
    }*/


    //printf("\noutput %i expected\n", it+1);
        //printf("%lf %lf %lf\n", courbe.parametres_expectes[0], courbe.parametres_expectes[1], courbe.parametres_expectes[2]);
        //printf("%lf %lf %lf\n", courbe.parametres_expectes_normalises[0], courbe.parametres_expectes_normalises[1], courbe.parametres_expectes_normalises[2]);

    /*
    for (int i=0; i<num_points; i++){
        printf("%lf\n", courbe.points[i]);
    }*/

    /*for (int i=0; i<iterations; i++){
        printf("%lf\n", erreurs_itertions[i]);
    }*/
