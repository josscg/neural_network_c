#ifndef COURBE_H
#define COURBE_H

typedef struct Courbe {
    int n_points;
    double *points;
    double *parametres_expectes;
    double *parametres_expectes_normalises;
}Courbe;

Courbe initialisation_courbe(int n_points);
void free_courbe(Courbe *courbe);

#endif
