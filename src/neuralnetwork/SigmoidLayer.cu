#include <neuralnetwork/SigmoidLayer.h>

SigmoidLayer::SigmoidLayer(std::string name)
{
  this->name = name;
}

Matrix SigmoidLayer::forward(const Matrix& X)
{
  Matrix res = X;
  Matrix e = X;

  for(int i = 0; i < res.getRows(); i++)
  {
    for(int j = 0; j < res.getCols(); j++)
    {
      if (res(i,j) > 10)
      {
        res(i,j) = 10;
      }
      else if (res(i, j) < -10)
      {
        res(i, j) = -10;
      }
      e(i, j) = exp(res(i,j));
      res(i, j) = e(i,j)+1;
      res(i, j) = e(i, j)/res(i,j);

    }
  }
  return res;
}

Matrix SigmoidLayer::backprop(Matrix& dZ, const Matrix& X, float learning_rate)
{
  // o(x)*(1-o(x))
  Matrix m1(X.getRows(), X.getCols(), 1.0f);
  Matrix ox = this->forward(dZ);
  m1 = m1 - ox;
  m1 = ox.hadamard(m1);
  return m1;
}