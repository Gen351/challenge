#pragma once
#include <sstream>   // <-- for ostringstream
#include <iomanip>   // <-- for setprecision

#include "../net.h"

#include <vector>
#include <string>

namespace NetworkSaver {
    std::string formatVectorDelimited(const std::vector<float>& values, char delim = ' ');
    std::string formatWeightSemicolonDelimited(const std::vector<std::vector<float>>& weight);
    inline std::vector<std::string> formatNetwork(const Net& net);

    /*
        1st line:   label line (layerCount, layerSizes[0].input.size(), layerSizes[0].output.size(), layerSizes[...layerCount - 1].output.size())
        2nd line:   layers[0].input (space delimited)\n
        3rd line:   layers[0].output (space delimited)\n
        4th line:   layers[0].bias (space delimited)\n
        5th line:   layers[0].gamma (space delimited)\n
        6th line:   layers[0].weight[0] (space delimited); layers[0].weight[1] (space delimited)\n
        ...
        ...
        ...
        n-4th line: layers[layerCount - 1].input (space delimited)\n
        n-3th line: layers[layerCount - 1].output (space delimited)\n
        n-2th line: layers[layerCount - 1].bias (space delimited)\n
        n-1th line: layers[layerCount - 1].gamma (space delimited)\n
        nth line:   layers[layerCount - 1].weight[0] (space delimited); layers[layerCount - 1].weight[1] (space delimited)   
    */

    inline std::vector<std::string> formatNetwork(const Net& net) {
        const std::vector<Layer>& networkLayers = net.getLayers(); 
        size_t layerCount = networkLayers.size();
        if(layerCount == 0) return {};

        std::vector<std::string> formattedNetwork;
        formattedNetwork.reserve(1 + layerCount * 5/*5 lines per layer*/);

        // LABEL LINE   
        std::ostringstream labelLine;
        // layerCount,
        labelLine << layerCount << ',';
        // layers[0].input.size(),
        labelLine << networkLayers[0].getLastInput().size() << ',';
        // layers[0...n-1].output.size(),
        for(size_t i = 0; i < layerCount; i++) {
            labelLine << networkLayers[i].getLastOutput().size();
            if(i + 1 != layerCount) labelLine << ',';
        }

        formattedNetwork.push_back(labelLine.str());

        // PER LAYER
        // layers[0..n-1]
        for(size_t i = 0; i < layerCount; i++) {
            const std::vector<float>& input  = networkLayers[i].getLastInput();
            const std::vector<float>& output = networkLayers[i].getLastOutput();
            const std::vector<float>& bias   = networkLayers[i].getBias();
            const std::vector<float>& gamma  = networkLayers[i].getGamma();
            const std::vector<std::vector<float>>& weight = networkLayers[i].getWeight();

            // format and push lines...
            formattedNetwork.push_back(formatVectorDelimited(input));
            formattedNetwork.push_back(formatVectorDelimited(output));
            formattedNetwork.push_back(formatVectorDelimited(bias));
            formattedNetwork.push_back(formatVectorDelimited(gamma));
            // weight
            formattedNetwork.push_back(formatWeightSemicolonDelimited(weight));
        }

        return formattedNetwork;
    }









































    
    std::string formatWeightSemicolonDelimited(const std::vector<std::vector<float>>& weight) {
        std::ostringstream line;

        size_t n = weight.size();
        for(size_t i = 0; i < weight.size(); i++) {
            line << formatVectorDelimited(weight[i]);
            if(i + 1 != n) line << ';';
        }

        return line.str();
    }

    std::string formatVectorDelimited(const std::vector<float>& values, char delim = ' ') {
        std::ostringstream line;
        line << std::setprecision(9);
        
        size_t n = values.size();
        for(size_t i = 0; i < n; i++) {
            line << values[i];
            if(i + 1 != n) line << delim;
        }

        return line.str();
    }
};
