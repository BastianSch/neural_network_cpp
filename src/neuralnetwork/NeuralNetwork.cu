#include <neuralnetwork/NeuralNetwork.h>
#include <neuralnetwork/LinearLayer.h>

void NeuralNetwork::addLayer(Layer* layer)
{
    layers.emplace_back(layer);
}

Matrix NeuralNetwork::forward(const Matrix& X)
{ 
    Matrix res = X;
    for (auto const& layer : layers)
    {
        res = layer->forward(res);
    }

    return res;
}

void NeuralNetwork::backward(Matrix& dY, const Matrix& X, float learning_rate)
{
    auto it = this->layers.rbegin();
    for ( ; it != this->layers.rend(); ++it)
    {
        dY = (*it)->backprop(dY, X, learning_rate);
    }   
}