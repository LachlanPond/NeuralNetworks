#include <stdint.h>
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYERS 2
#define HIDDEN_LAYER_SIZE 28
#define OUTPUT_LAYER_SIZE 10

typedef struct {
    float value;
    float bias;
    float *weights;
} nueron_t;

typedef struct {
    nueron_t inputLayer[INPUT_LAYER_SIZE];
    nueron_t hiddenLayers[HIDDEN_LAYERS][HIDDEN_LAYER_SIZE];
    nueron_t outputLayer[OUTPUT_LAYER_SIZE];
} network_t;

typedef struct {
    int32_t magicNumber;
    int32_t imageCount;
    int32_t rowCount;
    int32_t columnCount;
    uint8_t  *images;
} images_t;

typedef struct {
    int32_t magicNumber;
    int32_t itemCount;
    uint8_t  *labels;
} labels_t;

typedef struct {
    images_t imageData;
    labels_t labelData;
} dataset_t;