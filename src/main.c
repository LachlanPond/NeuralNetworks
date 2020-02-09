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
    images_t trainingImages;
    labels_t traingLabels;
    load_images("data/train-images.idx3-ubyte", &trainingImages);
    network_initialise(&network);
    free(trainingImages.images);
    return 0;
}