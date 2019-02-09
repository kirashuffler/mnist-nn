#ifndef NEURON_H
#define NEURON_H


class Neuron
{
protected:
    float data;
public:
    void set_data(float input);
    void calculate();
    float out();
};

#endif // NEURON_H
