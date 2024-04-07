#include <gtest/gtest.h>
#include <neuralnetwork/ReLULayer.h>
#include <neuralnetwork/SigmoidLayer.h>
#include <utils/Matrix.h>

namespace {

class BasicLayerTest : public ::testing::Test
{
protected:
  Matrix mat0;
  Matrix mat1;
  Matrix mat_1;

  ReLULayer layer_relu;
  SigmoidLayer layer_sigmoid;

public:
  BasicLayerTest(): mat0(3,3,0.0), mat1(3,3,1.0), mat_1(3,3,-1.0), layer_relu("relu"), layer_sigmoid("sigmoid"){};
};

TEST_F(BasicLayerTest, reluforward) {
  Matrix res = layer_relu.forward(mat0);
  EXPECT_EQ(res, mat0);
  res = layer_relu.forward(mat1);
  EXPECT_EQ(res, mat1);
  res = layer_relu.forward(mat_1);
  EXPECT_EQ(mat0, res);
}

TEST_F(BasicLayerTest, relubackward) {
  Matrix res = layer_relu.backprop(mat0, mat0, 1.0);
  EXPECT_EQ(res, mat0);
  res = layer_relu.backprop(mat1, mat1, 1.0);
  EXPECT_EQ(res, mat1);
  res = layer_relu.backprop(mat_1, mat_1, 1.0);
  EXPECT_EQ(mat0, res);
}

TEST_F(BasicLayerTest, sigmoidforward) {
  Matrix res = layer_sigmoid.forward(mat0);
  Matrix mat025(3, 3, 0.25);
  EXPECT_EQ(res, mat025);

  Matrix mate1(3,3,0.731059);
  res = layer_sigmoid.forward(mat1);
  EXPECT_EQ(res, mate1);
  
  Matrix mate_1(3,3,0.268941);
  res = layer_sigmoid.forward(mat_1);
  EXPECT_EQ(mate_1, res);
}

TEST_F(BasicLayerTest, sigmoidbackward) {
  Matrix mat05(3, 3, 0.5);
  Matrix res = layer_sigmoid.backprop(mat0, mat0, 1.0);
  EXPECT_EQ(res, mat05);
  res = layer_sigmoid.backprop(mat1, mat1, 1.0);
  EXPECT_EQ(res, mat1);
  res = layer_sigmoid.backprop(mat_1, mat_1, 1.0);
  EXPECT_EQ(mat0, res);
}

} // namespace

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}