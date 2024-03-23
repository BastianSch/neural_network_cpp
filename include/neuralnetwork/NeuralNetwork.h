#pragma once
#include <neuralnetwork/Layer.h>
#include <utils/Matrix.h>

#include <list>
#include <memory>

class NeuralNetwork{
private:
    std::list<Layer*> layers;
public:
    void addLayer(Layer* layer);
    Matrix forward(const Matrix& X);
    void backward(Matrix& dY, const Matrix& X, float learning_rate);
};