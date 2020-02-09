#ifndef INCLUDES
#define INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "../inc/header.h"
#include "../inc/functions.h"
#endif

int main() {
    network_t network;
    dataset_t trainingData;
    load_images("data/train-images.idx3-ubyte", &trainingData.imageData);
    load_labels("data/train-labels.idx1-ubyte", &trainingData.labelData);
    network_initialise(&network);
    free(trainingData.imageData.images);
    return 0;
}