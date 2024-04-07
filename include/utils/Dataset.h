# ifndef __DATASET_H__
# define __DATASET_H__

#include <string>
#include <vector>
#include <iostream>

struct DataPoint{
    std::vector<double> X;
    int Y;
};

class Dataset{
protected:
    std::vector<DataPoint> items;
    
public:
    virtual void loadFile(std::string path) = 0;
    DataPoint operator [](int index) { return items[index]; };
    int size(){ return items.size(); };
    virtual ~Dataset(){};
};

#endif