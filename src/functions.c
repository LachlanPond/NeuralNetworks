#ifndef INCLUDES
#define INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../inc/header.h"
#include "../inc/functions.h"
#endif

/* Initialize the network with randomized weights, neuron values, and biases */
void network_initialise(network_t *network) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        network->inputLayer[i].weights = (float*) malloc(HIDDEN_LAYER_SIZE * sizeof(float));
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            network->inputLayer[i].weights[j] = (float)(rand()) / (float)RAND_MAX;
        }
    }
    for (int hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++) {
        for (int neuron = 0; neuron < HIDDEN_LAYER_SIZE; neuron++) {
            network->hiddenLayers[hiddenLayer][neuron].value   = (float)(rand()) / (float)RAND_MAX;
            network->hiddenLayers[hiddenLayer][neuron].bias    = (float)(rand()) / (float)RAND_MAX;
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
            // Compute the value of the neuron that the loop is up to
            network->hiddenLayers[hiddenLayer][neuron].value = weightedSum + network->hiddenLayers[hiddenLayer][neuron].bias;
        }
    }
    // The output layer neuron values can be found by using the last of the hidden layer's neuron values
    for (int neuron = 0; neuron < OUTPUT_LAYER_SIZE; neuron++) {
        float weightedSum = 0.0f;
        for (int prevNeuron = 0; prevNeuron < HIDDEN_LAYER_SIZE; prevNeuron++) {
            weightedSum += network->hiddenLayers[HIDDEN_LAYERS-1][prevNeuron].weights[neuron] * network->hiddenLayers[HIDDEN_LAYERS-1][prevNeuron].value;
        }
        // Compute the value of the neuron that the loop is up to
        network->outputLayer[neuron].value = weightedSum + network->outputLayer[neuron].bias;
    }
}