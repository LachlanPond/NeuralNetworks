#ifndef INCLUDES
#define INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../inc/header.h"
#include "../inc/functions.h"
#endif

/* Initialize the network with randomized weights, neuron values, and biases */
void network_initialise(network_t *network) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        // The hiddedn layer size determines the amount of weights for each neuron of the input layer
        network->inputLayer[i].weights = (float*) malloc(HIDDEN_LAYER_SIZE * sizeof(float));
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            network->inputLayer[i].weights[j] = (float)(rand()) / (float)RAND_MAX;
        }
    }
    for (int hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++) {
        for (int neuron = 0; neuron < HIDDEN_LAYER_SIZE; neuron++) {
            network->hiddenLayers[hiddenLayer][neuron].value   = (float)(rand()) / (float)RAND_MAX;
            network->hiddenLayers[hiddenLayer][neuron].bias    = (float)(rand()) / (float)RAND_MAX;
            // Unless the hidden layer the loop is up to is the final hidden layer. Needed as the weight will relate to the output layer which is a different size
            if (hiddenLayer == HIDDEN_LAYERS-1) {
                network->hiddenLayers[hiddenLayer][neuron].weights = (float*) malloc(OUTPUT_LAYER_SIZE * sizeof(float));
                for (int weight = 0; weight < OUTPUT_LAYER_SIZE; weight++) {
                    network->hiddenLayers[hiddenLayer][neuron].weights[weight] = (float)(rand()) / (float)RAND_MAX;
                }
            }
            else {
                network->hiddenLayers[hiddenLayer][neuron].weights = (float*) malloc(HIDDEN_LAYER_SIZE * sizeof(float));
                for (int weight = 0; weight < HIDDEN_LAYER_SIZE; weight++) {
                    network->hiddenLayers[hiddenLayer][neuron].weights[weight] = (float)(rand()) / (float)RAND_MAX;
                }
            }
            
        }
    }
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        network->outputLayer[i].value   = (float)(rand()) / (float)RAND_MAX;
        network->outputLayer[i].bias    = (float)(rand()) / (float)RAND_MAX;
    }
}

/* Propogate through the whole network */
void propogate_forward(network_t *network) {
    /* Need to find the values of the nerons for each layer after the input layer */
    // Start by looping through the hidden layers
    for (int hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++) {
        // The weighted sum of the previous layer's neuron values must be found to find the value of each neruon in the current layer
        for (int neuron = 0; neuron < HIDDEN_LAYER_SIZE; neuron++) {
            float weightedSum = 0.0f;
            // If the we are find the neuron values for the first of the hidden layers, we need to use the input layers neurons
            if (hiddenLayer == 0) {
                for (int prevNeuron = 0; prevNeuron < INPUT_LAYER_SIZE; prevNeuron++) {
                    weightedSum += network->inputLayer[prevNeuron].weights[neuron] * network->inputLayer[prevNeuron].value;
                }
            }
            // For the rest of the hidden layers, the previous hidden layer can be used
            else {
                for (int prevNeuron = 0; prevNeuron < HIDDEN_LAYER_SIZE; prevNeuron++) {
                    weightedSum += network->hiddenLayers[hiddenLayer-1][prevNeuron].weights[neuron] * network->hiddenLayers[hiddenLayer-1][prevNeuron].value;
                }
            }
            // Compute the value of the neuron that the loop is up to using the ReLU of the weighted sum + the bias
            network->hiddenLayers[hiddenLayer][neuron].value = ReLU(weightedSum + network->hiddenLayers[hiddenLayer][neuron].bias);
        }
    }
    // The output layer neuron values can be found by using the last of the hidden layer's neuron values
    for (int neuron = 0; neuron < OUTPUT_LAYER_SIZE; neuron++) {
        float weightedSum = 0.0f;
        for (int prevNeuron = 0; prevNeuron < HIDDEN_LAYER_SIZE; prevNeuron++) {
            weightedSum += network->hiddenLayers[HIDDEN_LAYERS-1][prevNeuron].weights[neuron] * network->hiddenLayers[HIDDEN_LAYERS-1][prevNeuron].value;
        }
        // Compute the value of the neuron that the loop is up to using the ReLU of the weighted sum + the bias
        network->outputLayer[neuron].value = ReLU(weightedSum + network->outputLayer[neuron].bias);
    }
}

float ReLU(float input) {
    if (0.0f > input) return 0.0f;
    else return input;
}

void load_images(char *fileLocation, images_t *imageSet) {
    FILE *fp;
    uint32_t fileLength = 0;
    uint8_t *buffer;
    fp = fopen(fileLocation, "rb");   // Open file in binary mode
    if (fp == NULL) {
        perror("Error in opening file");
    }
    do {
        fgetc(fp);
        fileLength++;
    } while (!feof(fp));
    rewind(fp);

    buffer = (uint8_t*) malloc(fileLength * sizeof(uint8_t));
    fileLength = 0;
    do {
        buffer[fileLength] = fgetc(fp);
        fileLength++;
    } while (!feof(fp));
    fclose(fp);

    imageSet->images = (uint8_t *) malloc((fileLength - 16) * sizeof(uint8_t));
    // Populate the image set struct
    imageSet->magicNumber   = (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + (buffer[3] << 0);
    imageSet->imageCount    = (buffer[4] << 24) + (buffer[5] << 16) + (buffer[6] << 8) + (buffer[7] << 0);
    imageSet->rowCount      = (buffer[8] << 24) + (buffer[9] << 16) + (buffer[10] << 8) + (buffer[11] << 0);
    imageSet->columnCount   = (buffer[12] << 24) + (buffer[13] << 16) + (buffer[14] << 8) + (buffer[15] << 0);
    memcpy(imageSet->images, &buffer[16], (fileLength - 16) * sizeof(uint8_t));
    free(buffer);
}

void load_labels(char *fileLocation, labels_t *labelSet) {
    FILE *fp;
    uint32_t fileLength = 0;
    uint8_t *buffer;
    fp = fopen(fileLocation, "rb");   // Open file in binary mode
    if (fp == NULL) {
        perror("Error in opening file");
    }
    do {
        fgetc(fp);
        fileLength++;
    } while (!feof(fp));
    rewind(fp);

    buffer = (uint8_t*) malloc(fileLength * sizeof(uint8_t));
    fileLength = 0;
    do {
        buffer[fileLength] = fgetc(fp);
        fileLength++;
    } while (!feof(fp));
    fclose(fp);

    labelSet->labels = (uint8_t *) malloc((fileLength - 8) * sizeof(uint8_t));
    // Populate the label set struct
    labelSet->magicNumber   = (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + (buffer[3] << 0);
    labelSet->itemCount     = (buffer[4] << 24) + (buffer[5] << 16) + (buffer[6] << 8) + (buffer[7] << 0);
    memcpy(labelSet->labels, &buffer[8], (fileLength - 8) * sizeof(uint8_t));
    free(buffer);
}