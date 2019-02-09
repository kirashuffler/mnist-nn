#include "derivative.h"
float Derivative(float arg){
    arg = arg * (1 - arg);
    return arg;
}
