#pragma once
#include <sstream>   // <-- for ostringstream
#include <iomanip>   // <-- for setprecision

#include "../net.h"

#include <vector>
#include <string>

#include "dataset_loader.h"

namespace NetworkLoader {

    // --- Helper: Split "0.1 0.5 0.9" into vector<float> ---
    inline std::vector<float> parseVector(const std::string& line, char delim = ' ') {
        std::vector<float> values;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, delim)) {
            if (!token.empty()) {
                values.push_back(std::stof(token));
            }
        }
        return values;
    }

    // --- Helper: Split "0.1 0.2; 0.3 0.4" into vector<vector<float>> ---
    inline std::vector<std::vector<float>> parseWeightMatrix(const std::string& line) {
        std::vector<std::vector<float>> matrix;
        std::stringstream ss(line);
        std::string rowString;
        
        // 1. Split by semicolon (Rows)
        while (std::getline(ss, rowString, ';')) {
            if (!rowString.empty()) {
                // 2. Split by space (Columns) inside the row
                matrix.push_back(parseVector(rowString, ' '));
            }
        }
        return matrix;
    }

    inline Net load(const std::string& networkFilePath) {
        // 1. Read all lines from file
        const std::vector<std::string> lines = DatasetLoader::load(networkFilePath);
        
        if (lines.empty()) {
            throw std::runtime_error("Network file is empty: " + networkFilePath);
        }

        // 2. Parse Header (Line 0)
        // Format: layerCount, inputSize, outputSize0, outputSize1...
        // Example: "2,784,64,10"
        std::vector<float> headerParams = parseVector(lines[0], ',');
        
        size_t layerCount = (size_t)headerParams[0];
        int inputSize = (int)headerParams[1];

        // Reconstruct Topology: {Input, Hidden..., Output}
        std::vector<int> topology;
        topology.push_back(inputSize); 
        
        for(size_t i = 0; i < layerCount; i++) {
            // headerParams starts at index 2 for the layer outputs
            topology.push_back((int)headerParams[2 + i]);
        }

        // 3. Create the Net (initially random weights)
        Net net(topology);
        
        // We need access to the layers to overwrite them
        // Make sure your Net class has a method: std::vector<Layer>& getLayersMutable(); 
        // OR simply: const_cast<std::vector<Layer>&>(net.getLayers()); if you are lazy.
        std::vector<Layer>& layers = const_cast<std::vector<Layer>&>(net.getLayers());

        // 4. Fill Weights and Biases
        // The file structure is: Header, then [Bias, Weight] for each layer.
        size_t currentLine = 1;

        for (size_t i = 0; i < layerCount; i++) {
            if (currentLine + 1 >= lines.size()) break;

            // Load Bias (Space delimited)
            std::vector<float> bias = parseVector(lines[currentLine]);
            layers[i].setBias(bias); 
            currentLine++;

            // Load Weight (Semicolon delimited rows, Space delimited cols)
            std::vector<std::vector<float>> weight = parseWeightMatrix(lines[currentLine]);
            layers[i].setWeight(weight);
            currentLine++;
        }

        std::cout << "Network successfully loaded from " << networkFilePath << "\n";
        return net;
    }


};