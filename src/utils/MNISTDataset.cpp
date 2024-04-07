#include <utils/MNISTDataset.h>
#include <string>
#include <fstream>
#include <iostream>

void MNISTDataset::loadFile(std::string path)
{
    std::string line;
    std::ifstream data_file;
    std::string delim = ",";

    data_file.open(path);
    if (data_file.is_open())
    {
        while (std::getline(data_file, line))
        {
            int start = 2; // first datapoint is y
            int end = line.find(delim);
            DataPoint datapoint = {};
            datapoint.Y = static_cast<double>(line[0] - '0');
            while (end != std::string::npos)
            {
                datapoint.X.push_back((float)std::stoi(line.substr(start, end - start))/255.0f);
                start = end + delim.length();
                end = line.find(delim, start);
            }
            items.push_back(datapoint);
        }
        data_file.close();
    }
    else
    {
        std::cout << "Unable to open file " << path << std::endl;
    }
}