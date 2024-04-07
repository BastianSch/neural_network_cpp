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
        
        std::cout << layer->getName() << " " << res << "\n";
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

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn)
{
    os << "Neural Network:\n{\n";
    for (auto const& layer : nn.layers)
    {
        os << "\t[" << layer->getName() << "]\n";
    }
    os << "}" << std::endl;
    return os;
}