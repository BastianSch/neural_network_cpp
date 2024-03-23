#pragma once
#include <neuralnetwork/Layer.h>
#include <utils/Matrix.h>

class ReLULayer : public Layer {
public:
    ReLULayer(std::string name);
    Matrix forward(const Matrix& X);
    Matrix backprop(Matrix& dZ, const Matrix& X, float learning_rate);
};

