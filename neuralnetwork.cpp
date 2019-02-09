#include "neuralnetwork.h"
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include "mnistreader.h"
#include "derivative.h"

NeuralNetwork::NeuralNetwork(unsigned int inp, unsigned int hid, unsigned int out)
{
    input = inp;
    hidden = hid;
    output = out;
    inputLayer.resize(input);
    hiddenLayer.resize(hidden);
    outputLayer.resize(output);
    in_bias.resize(hidden);
    hid_bias.resize(output);
    hid_Weights.resize(hidden, std::vector<float> (input));
    out_Weights.resize(output, std::vector<float> (hidden));
}
//random filling input-hidden neuron layers connection
void NeuralNetwork::weights_init(){
    float scale_factor = 0.7 * pow(hidden, 1 / input);
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_real_distribution<float> distribution (-0.5, 0.5);
    for (size_t i = 0; i < hidden; ++i)
        for (size_t j = 0; j < input; ++j)
            hid_Weights[i][j] = distribution(generator);
//correcting input weights
    float lengthes[hidden];
    for (size_t i = 0; i < hidden; ++i){
        float length = 0;
        for (size_t j = 0; j < input; ++j)
            length += pow(hid_Weights[i][j], 2);
        length = sqrt(length);
        lengthes[i] = length;
    }

    for (size_t i = 0; i < hidden; ++i)
        for (size_t j = 0; j < input; ++j)
            hid_Weights[i][j] = scale_factor * hid_Weights[i][j] / lengthes[i];
//initialize input bias with values between -scale_factor and +scale_factor;
    std::uniform_real_distribution<float> dist(-scale_factor, scale_factor);
    for(size_t i = 0; i < hidden; ++i){
        in_bias[i] = dist(generator);
    }
//random filling hidden-output neuron connections connections
    for (size_t i = 0; i < output; ++i)
        for (size_t j = 0; j < hidden; ++j)
            out_Weights[i][j] = distribution(generator);
//random filling hidden bias with values between -0.5 and 0.5
    for (size_t i = 0; i < output; ++i)
        hid_bias[i] = distribution(generator);
}


void NeuralNetwork::save_to_file(char* Directory){
    std::ofstream fout;
    fout.open(Directory);
    for (size_t i = 0; i < hidden; ++i){
        for (size_t j = 0; j < input; ++j)
            fout << hid_Weights[i][j] << ' ';
        fout << '\n';
    }
    fout << '\n';
    for (size_t i = 0; i < hidden; ++i)
        fout << hid_bias[i] << ' ';
    fout << '\n';
    for (size_t i = 0; i < output; ++i){
        for (size_t j = 0; j < hidden; ++j)
            fout << out_Weights[i][j] << ' ';
        fout << '\n';
    }
    fout << '\n';
    for (size_t i = 0; i < output; ++i)
        fout << hid_bias[i] << ' ';
    fout << '\n';
    fout.close();
}

void NeuralNetwork::load_from_file(char* Directory){
    std::ifstream fin;
    fin.open(Directory);
    float x;
    for (size_t i = 0; i < hidden; ++i)
        for (size_t j = 0; j < input; ++j){
            fin >> x;
            hid_Weights[i][j] = x;
        }
    for (size_t i = 0; i < hidden; ++i){
        fin >> x;
        in_bias[i] = x;
    }
    for (size_t i = 0; i < output; ++i)
        for (size_t j = 0; j < hidden; ++j){
            fin >> x;
            out_Weights[i][j] = x;
        }
    for (size_t i = 0; i < output; ++i){
        fin >> x;
        hid_bias[i] = x;
    }
    fin.close();
}

void NeuralNetwork::process(){
    //transfering data to hidden layer and calculation
    for (size_t i = 0; i < hidden; ++i){
        float sum = 0;
        for (size_t j = 0; j < input; ++j)
            sum += inputLayer[j].out() * hid_Weights[i][j];
        sum += in_bias[i];
        hiddenLayer[i].set_data(sum);
        hiddenLayer[i].calculate();
    }

    //transfering data to output layer and calculation
    for (size_t i = 0; i < output; ++i){
        float sum = 0;
        for (size_t j = 0; j < hidden; ++j)
            sum += hiddenLayer[j].out() * out_Weights[i][j];
        outputLayer[i].set_data(sum);
        outputLayer[i].calculate();
    }
}

void NeuralNetwork::learn(unsigned int iterations, char* imageDirectory, char* labelDirectory, char* weightsDirectory){
    std::cout << "copying\n";
    std::vector< std::vector<float> > examples;
    std::vector<float> targets;
    int DataSize = 0, magic = 0, columns = 0, rows = 0;
    std::ifstream file;
    file.open(imageDirectory, std::ios::binary);
    MNISTReader(file, examples, magic, DataSize, rows, columns);
    file.close();
    file.open(labelDirectory, std::ios::binary);
    MNISTReader(file, targets, magic, DataSize);
    file.close();
    std::cout << "copying finished\n";

    float coef = 0.5;

    //learning
    for (size_t iteration = 0; iteration < iterations; ++iteration){
        for (size_t it = 0; it < DataSize; ++it){
            //putting learning data
            for (unsigned int i = 0; i < input; ++i)
                inputLayer[i].set_data(examples[it][i]);
            process();
            std::vector<float> delta_in_bias(hidden);
            std::vector<float> delta_hid_bias(output);
            std::vector<float> delta_out(output);
            std::vector<float> delta_hid(hidden);
            std::vector<float> targets_prototype(output, 0);
            targets_prototype[round(targets[it])] = 1;
            Weights delta_hid_out;
            Weights delta_in_hid;
            delta_in_hid.resize(hidden, std::vector<float> (input));
            delta_hid_out.resize(output, std::vector<float> (hidden));
            for (size_t i = 0; i < output; ++i)
                delta_out[i] = Derivative(outputLayer[i].out()) * (targets_prototype[i] - outputLayer[i].out());

            for (size_t i = 0; i < output; ++i)
                for(size_t j = 0; j < hidden; ++j){
                    float delta = coef * delta_out[i] * hiddenLayer[j].out();
                    delta_hid_out[i][j] = delta;
            }
            //calculatinc values for hidden bias correction
            for (size_t i = 0; i < output; ++i){
                float delta = coef * delta_out[i];
                delta_hid_bias[i] = delta;
            }
            //calculating hidden layers errors
            for (size_t i = 0; i < hidden; ++i){
                float sum = 0;
                for (size_t j = 0; j < output; ++j)
                    sum += delta_out[j] * out_Weights[j][i];
                delta_hid[i] = Derivative(hiddenLayer[i].out()) * sum;
            }

            // calculating values for correction input to hidden weights
            for (size_t i = 0; i < hidden; ++i)
                for (size_t j = 0; j < input; ++j){
                    float delta = coef * delta_hid[i] * inputLayer[j].out();
                    delta_in_hid[i][j] = delta;
                }
            //calculating values for correction input bias
            for (size_t i = 0; i < hidden; ++i){
                float delta = coef * delta_hid[i];
                delta_in_bias[i] = delta;
            }

            //correction of all the weights and biases
            for (size_t i = 0; i < output; ++i){
                for (size_t j = 0; j < hidden; ++j)
                    out_Weights[i][j] += delta_hid_out[i][j];
                hid_bias[i] += delta_hid_bias[i];
            }
            for (size_t i = 0; i < hidden; ++i){
                for (size_t j = 0; j < input; ++j)
                    hid_Weights[i][j] += delta_in_hid[i][j];
                in_bias[i] = delta_in_bias[i];
            }
        }
        std::cout << "iterations: " << iteration;
    }
    save_to_file(weightsDirectory);
}

void NeuralNetwork::test(char* imageDirectory, char* labelDirectory){
    std::vector<std::vector<float> > test_images;
    std::vector<float> test_labels;
    int magic = 0, size = 0, rows = 0, columns = 0;
    std::ifstream file;
    file.open(imageDirectory, std::ios::binary);
    MNISTReader(file, test_images, magic, size, rows, columns);
    file.close();
    file.open(labelDirectory, std::ios::binary);
    MNISTReader(file, test_labels, magic, size);
    unsigned s = 0;
    std::ofstream fout;
    fout.open("result.txt");
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < rows; ++j){
            for (size_t k = 0; k < columns; ++k)
               fout << (char)(255*test_images[i][j*columns + k]) << " ";
            fout << "\n";
        }
        float labels_answer = test_labels[i];
        fout << "Answer is: " << labels_answer << ";\n";
        for (int j = 0; j < input; ++j)
            inputLayer[j].set_data(test_images[i][j]);
        process();
        float max = outputLayer[0].out(), max_index = 0, sum = 0;
        for (size_t i = 1; i < output; ++i)
            if (max < outputLayer[i].out()){
                sum += outputLayer[i].out();
                max = outputLayer[i].out();
                max_index = i;
            }
        float NNAnswer = max_index;
        fout << "NN's answer is: " << NNAnswer << ";\n";
        if (NNAnswer != labels_answer)
            s++;
    }
    fout << "Errors:" << s;
    std::cout << "Precision is:" << (1 - ((float) s / (float) size)) * 100 << "%\n";
    fout.close();
}
