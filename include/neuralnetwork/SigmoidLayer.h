#pragma once
#include <neuralnetwork/Layer.h>
#include <utils/Matrix.h>

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(std::string name);
    Matrix forward(const Matrix& X);
    Matrix backprop(Matrix& dZ, const Matrix& X, float learning_rate);
};

