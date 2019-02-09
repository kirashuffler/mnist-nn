#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "neuron.h"
#include <vector>

typedef std::vector<Neuron> NeuralLayer;
typedef std::vector<std::vector<float> > Weights;
class NeuralNetwork
{
protected:
    unsigned int input;
    unsigned int hidden;
    unsigned int output;
    std::vector<float> in_bias;
    std::vector<float> hid_bias;
    NeuralLayer inputLayer;
    Weights hid_Weights;
    NeuralLayer hiddenLayer;
    Weights out_Weights;
    NeuralLayer outputLayer;
public:
    NeuralNetwork(unsigned int inp,unsigned int hid, unsigned int out);
    void weights_init();
   // void read_data(char* FileDirectory);
    void save_to_file(char* Directory);
    void load_from_file(char* Directory);
    void process();
    void learn(unsigned int iterations, char* imageDirectory, char* labelDirectory, char* weightsDirectory);
    void test(char* imageDirectory, char* labelDirectory);
};

#endif // NEURALNETWORK_H
