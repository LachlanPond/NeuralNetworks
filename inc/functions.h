void network_initialise(network_t *network);
void propogate_forward(network_t *network);
float ReLU(float input);
void load_images(char *fileLocation, images_t *imageSet);