#include "neuron.h"
#include <cmath>
void Neuron::set_data(float input){
    data = input;
}

void Neuron::calculate()
{
    data = 1 / (1 + exp(-data));
}

float Neuron::out() {
    return data;
}
