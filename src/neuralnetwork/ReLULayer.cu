#include <neuralnetwork/ReLULayer.h>


ReLULayer::ReLULayer(std::string name)
{
  this->name = name;
}

Matrix ReLULayer::forward(const Matrix& X)
{
  Matrix res = X;
  for(int i = 0; i < res.getRows(); i++)
  {
    for(int j = 0; j < res.getCols(); j++)
    {
      if(res(i, j)<0)
        {res(i, j)=0.0;}
    }
  }
  return res;
}

Matrix ReLULayer::backprop(Matrix& dZ, const Matrix& X, float learning_rate)
{
  for(int i = 0; i < dZ.getRows(); i++)
    {
      for(int j = 0; j < dZ.getCols(); j++)
      {
        if(dZ(i, j)<=0)
          {dZ(i, j)=0.0;}
        else
          {dZ(i, j)=1.0;}
      }
    }
    
    return dZ;
}