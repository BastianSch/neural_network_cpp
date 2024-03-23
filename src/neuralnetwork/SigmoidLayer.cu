#include <neuralnetwork/SigmoidLayer.h>

SigmoidLayer::SigmoidLayer(std::string name)
{
  this->name = name;
}

Matrix SigmoidLayer::forward(const Matrix& X)
{
  Matrix res = X;
    for(int i = 0; i < res.getRows(); i++)
    {
      for(int j = 0; j < res.getCols(); j++)
      {
        res(i, j) = 1/(1+exp(-res(i,j)));
      }
    }
    return X;
}

Matrix SigmoidLayer::backprop(Matrix& dZ, const Matrix& X, float learning_rate)
{
  // o(x)*(1-o(x))
  Matrix m1(X.getRows(), X.getCols(), 1.0f);
  Matrix ox = this->forward(dZ);
  m1 = m1 - ox;
  Matrix ox_T = ox.transpose();
  m1 = ox_T * m1;

  return m1.transpose();
}