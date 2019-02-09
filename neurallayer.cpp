#include "neurallayer.h"

NeuralLayer::NeuralLayer()
{
}

NeuralLayer::SetSize(unsigned int s){
    size = s;
    array = new Neuron[size];
}

NeuralLayer::~NeuralLayer(){
    delete[] array;
}

Neuron* NeuralLayer::ptr(){
    return array;
}
