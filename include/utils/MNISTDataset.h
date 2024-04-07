# ifndef __MNISTDATASET_H__
# define __MNISTDATASET_H__

#include <utils/Dataset.h>

#include <string>
#include <vector>

class MNISTDataset : public Dataset {    
public:
    void loadFile(std::string path);
};

#endif