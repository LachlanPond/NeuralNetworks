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
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        network.inputLayer[i].value = trainingData.imageData.images[i];
    }
    propogate_forward(&network);
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        printf("%f ", network.outputLayer[i].value);
    }
    printf("\n");
    free(trainingData.imageData.images);
    return 0;
}