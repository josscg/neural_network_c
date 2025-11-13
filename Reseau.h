#ifndef RESEAU_H
#define RESEAU_H
#include "Couche.h"

typedef struct Reseau {
    int n_couches;
    Couche *couches;
}Reseau;

Reseau initialisation_reseau(int n_couches);
void free_reseau(Reseau *reseau);

#endif
