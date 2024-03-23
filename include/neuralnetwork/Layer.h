#pragma once
#include <iostream>
#include <utils/Matrix.h>

class Layer {

protected:
    std::string name;

public:
    virtual Matrix forward(const Matrix& X) = 0;
    virtual Matrix backprop(Matrix& dZ, const Matrix& X, float learning_rate) = 0;

    std::string getName() { return this->name; };
    virtual ~Layer(){};
};
