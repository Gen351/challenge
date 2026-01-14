#pragma once
#include<vector>
#include<fstream>
#include<string>

#include<stdexcept>

namespace DatasetLoader {
    inline std::vector<std::string> load(const std::string& datasetPath) {
        std::ifstream read(datasetPath);
        if(!read) {
            throw std::runtime_error(datasetPath + " does not exist!");
        }

        std::vector<std::string> datasetFormat;
        std::string line;
        while(getline(read, line)) {
            datasetFormat.push_back(line);
        }
        read.close();
        
        return datasetFormat;
    }

};