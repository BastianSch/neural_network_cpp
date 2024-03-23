#pragma once
#include <neuralnetwork/Layer.h>
#include <utils/Matrix.h>

class LinearLayer : public Layer {
private:
    Matrix* W;
    Matrix* b;
public:
    LinearLayer(int input_length, int output_length, std::string name);
    Matrix forward(const Matrix& X);
    Matrix backprop(Matrix& dZ, const Matrix& X, float learning_rate);
};

