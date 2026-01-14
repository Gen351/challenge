#pragma once
#include<iostream>

#include "../net.h"

#include<vector>
#include<string>


class NetworkSaver {
    // to prevent instantiation
    NetworkSaver() = delete;
    
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

    std::vector<std::string> formatNetwork(const Net& net) {
        const std::vector<Layer> networkLayers = net.getLayers(); 
        
        int layerCount = networkLayers.size();
        
        // layerCount,
        std::string labelLine = std::to_string(layerCount) + ',';
        // layers[0].input
        labelLine += std::to_string(networkLayers[0].getLastIntput().size()) + ',';
        // layers[0...n-1].output
        for(int i = 0; i < layerCount; i++) {
            labelLine += std::to_string(networkLayers[i].getLastOutput().size()) + ',';
        }
        labelLine += '\n';
    }

};