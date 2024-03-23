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
  //std::cout << "forward\n" << this->getName() << "\nX:\n" << X << "\nW:\n" << *this->W << std::endl;
  //std::cout << "X:\n" << X;
  //std::cout << "W:\n" << *this->W;
  //Matrix res(X.getRows(), X.getCols(), 0.0f);
  Matrix res = X * *this->W;
  //std::cout << "Res1:\n" << res;

  res = res + *this->b;
  //std::cout << "Res OUT:\n" << res;

  return res;
}

Matrix LinearLayer::backprop(Matrix& dZ, const Matrix& X, float learning_rate)
{
  //std::cout << "backpropagation\n" << this->getName() << "\nX:\n" << X << "\nW:\n" << *this->W << std::endl;

  /* W = W - learning_rate * d_loss/dw
   * d_loss/dw = dloss/dy(=Loss) * dy/dw
   * dy/dw -> x */

  //std::cout<< "Weights before: " << std::endl << *this->W;

  //std::cout << "X\n" << X << "\ndZ\n" << dZ << std::endl;
  Matrix X_T = X;
  X_T = X_T.transpose();
  //std::cout << "X_T\n" << X_T << std::endl;

  Matrix dw = dZ * X_T; // -> 
  //std::cout<< "dw" << std::endl << dw << std::endl;
  //std::cout<< "W" << std::endl << *this->W << std::endl;
  *this->W = *this->W - (learning_rate * dw);
  //std::cout<< "W" << std::endl << *this->W;
  //std::cout<< "dZ" << std::endl << dZ;

  // Derivative
  Matrix res = *this->W * dZ;
  
  return res;
}
