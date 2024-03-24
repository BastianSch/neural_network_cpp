#include <neuralnetwork/NeuralNetwork.h>
#include <neuralnetwork/LinearLayer.h>
#include <neuralnetwork/ReLULayer.h>
#include <neuralnetwork/SigmoidLayer.h>

#include <utils/MNISTDataset.h>

#include <vector>

int main(int argc, char* argv[])
{
  const int input_length = 4, output_length = 1;
  const int num_epochs = 1000000;

  NeuralNetwork net;
  MNISTDataset train_dataset;
  std::string data_path = "data/mnist_train.csv"; 
  train_dataset.loadFile(data_path);

  DataPoint p = train_dataset[0];


  int num_layers = 10;

  for(int i = 0; i<num_layers; i++)
  {
    net.addLayer(new LinearLayer(input_length, input_length, "LinearLayer_"+std::to_string(i)));
    if(i<num_layers-1)
    {net.addLayer(new ReLULayer("ReLULayer_"+std::to_string(i)));}
    else
    {
      net.addLayer(new LinearLayer(input_length, output_length, "LinearLayer_"+std::to_string(i)));
      net.addLayer(new SigmoidLayer("SigmoidLayer_"+std::to_string(i)));
    }
  }


  Matrix Y_(1, input_length, 0.0f);
  Matrix dY(1, input_length, 0.0f);

  float loss1 = 0.0f;
  float loss2 = 0.0f;

  for(int i = 0; i < num_epochs; i++)
  {
    loss1 = 0.0f;
    //std::cout << "X1:\n" << X1 << std::endl;
    Y_ = net.forward(X1);
    //std::cout << "Y_:\n" << Y_;
    //std::cout << "Y_pred:\n" << Y_ << "Y:\n" << Y << "############\n";

    //std::cout << "Calculate dY" << std::endl;
    dY = Y_ - Y1;
    for(int j = 0; j < dY.getRows(); j++)
    {
      for(int k = 0; k < dY.getCols(); k++)
      {
        loss1+=(dY(j, k)*dY(j, k));
      }
    }  
    //std::cout << "X1:" << std::endl << X1 << std::endl;
    net.backward(dY, X1, 0.01);
       
    Y_ = net.forward(X2);

    dY = Y_ - Y2;
    //std::cout << "dY:" << std::endl << dY << std::endl;
    loss2 = 0.0f;
    for(int j = 0; j < dY.getRows(); j++)
    {
      for(int k = 0; k < dY.getCols(); k++)
      {
        loss2+=(dY(j,k)*dY(j,k));
      }
    }  
    std::cout << std::to_string(loss1) << "\t" << std::to_string(loss2) << "\t" << std::to_string(loss1 += loss2) << std::endl;
    net.backward(dY, X2, 0.01);
    
  }
  return 0;
}
