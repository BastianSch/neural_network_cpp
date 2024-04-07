#include <neuralnetwork/LinearLayer.h>
#include <cassert>

LinearLayer::LinearLayer(int input_length, int output_length, std::string name)
{
  this->name = name;
  this->W = new Matrix(input_length, output_length, true);
  this->b = new Matrix(1, output_length, true);
}

Matrix LinearLayer::forward(const Matrix& X)
{
  Matrix res = X * *this->W;

  res = res + *this->b;

  return res;
}

Matrix LinearLayer::backprop(Matrix& dZ, const Matrix& X, float learning_rate)
{
  /* W = W - learning_rate * d_loss/dw
   * d_loss/dw = dloss/dy(=Loss) * dy/dw
   * dy/dw -> x */

  // dZ [1,10], X[1,784], W[784, 10]
  Matrix dZ_T = dZ; // [1,10]
  dZ_T = dZ_T.transpose(); // [10, 1]
  
  Matrix dW = dZ_T * X; // [10, 1] * [1, 784] -> [10, 784] 

  dW = learning_rate * dW;
  dW = dW.transpose();

  *this->W = *this->W - dW;

  // Derivative
  Matrix res = dZ * *this->W; // [1, 10] * [10, 784] -> [1, 784]
  return res;
}
