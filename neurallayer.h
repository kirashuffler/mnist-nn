#ifndef NEURALLAYER_H
#define NEURALLAYER_H
#include "neuron.h"

class NeuralLayer
{
protected:
    unsigned int size;
    Neuron* array;
public:
    NeuralLayer();
    SetSize(unsigned int s);
    ~NeuralLayer();
    Neuron* ptr();
};

#endif // NEURALLAYER_H
