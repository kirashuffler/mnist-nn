#include <iostream>
#include "neuralnetwork.h"
using namespace std;
int main(){
    NeuralNetwork NN(784, 90, 10);
    char input;
    cout << "Press 'L' for learning, 'T' - for testing\n";
    cin >> input;
    if (input == 'L'){
        unsigned iterations;
        cout << "Enter the number of iterations:\n";
        cin >> iterations;
        NN.weights_init();
        NN.learn(iterations, "data/Train images", "data/Train labels", "data/NeuralNetwork.dat");
    }else if (input == 'T'){
        NN.load_from_file("data/NeuralNetwork.dat");
        NN.test("data/Test images", "data/Test labels");
    }
    return 0;
}
