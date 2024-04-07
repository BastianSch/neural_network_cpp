#include <neuralnetwork/NeuralNetwork.h>
#include <neuralnetwork/LinearLayer.h>
#include <neuralnetwork/ReLULayer.h>
#include <neuralnetwork/SigmoidLayer.h>

#include <utils/MNISTDataset.h>

#include <vector>

int main(int argc, char* argv[])
{
  const int input_length = 784, output_length = 10;
  const int num_epochs = 100;
  const float learning_rate = 0.0001;

  NeuralNetwork net;
  MNISTDataset train_dataset;
  std::string data_path = "data/mnist_train.csv"; 
  train_dataset.loadFile(data_path);
  std::cout << "Loaded Dataset: Length " << train_dataset.size() << std::endl;

  std::cout << "First Entry: " << train_dataset[0].Y << std::endl;

  int num_layers = 3;

  for(int i = 0; i<num_layers; i++)
  {
    if(i<num_layers-1)
    {
      net.addLayer(new LinearLayer(input_length, input_length, "LinearLayer_"+std::to_string(i)));
      net.addLayer(new ReLULayer("ReLULayer_"+std::to_string(i)));
    }
    else
    {
      net.addLayer(new LinearLayer(input_length, output_length, "LinearLayer_"+std::to_string(i)));
      net.addLayer(new SigmoidLayer("SigmoidLayer_"+std::to_string(i)));
    }
  }
  std::cout << "Initialized Neural Net" << std::endl;
  std::cout << net << std::endl;



  Matrix Y_(1, 10, 0.0f);
  Matrix Y(1, 10, 0.0f);
  Matrix dY(1, 10, 0.0f);

  float loss_epoch = 0.0f;
  float loss_iteration = 0.0f;

  for(int epoch = 0; epoch < num_epochs; epoch++)
  {
    loss_epoch = 0.0f;
    DataPoint datapoint;
    Matrix X(1, 784);
    
    for(int it = 0; it < train_dataset.size(); it++)
    {
      loss_iteration = 0.0f;
      datapoint = train_dataset[it];
      std::vector<std::vector<double>> vec_x;
      vec_x.push_back(datapoint.X);

      X.setValues(vec_x);
      Y_ = net.forward(X);

      std::vector<std::vector<double>> vec_y;
      std::vector<double> y_onehot = {0,0,0,0,0,0,0,0,0,0};
      y_onehot[datapoint.Y] = 1.0f;

      vec_y.push_back(y_onehot);
      Y.setValues(vec_y);

      std::cout << "Y " << datapoint.Y << std::endl;

      dY = Y_ - Y;

      for(int j = 0; j < dY.getRows(); j++)
      {
        for(int k = 0; k < dY.getCols(); k++)
        {
          loss_iteration+=(dY(j, k)*dY(j, k));
        }
      }  
      std::cout << "Y " << Y << " Y_ " << Y_ << " " << loss_iteration << std::endl;
      loss_epoch+=loss_iteration;

      net.backward(dY, X, learning_rate);
    }
    std::cout << "Y_ " << Y_ << "\n" << "Y " << Y << std::endl;
    loss_iteration = loss_iteration/train_dataset.size();
    std::cout << epoch << ": " << std::to_string(loss_iteration) << std::endl;
  }
  return 0;
}
