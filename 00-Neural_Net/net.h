#pragma once

#include<vector>
#include<stdexcept>

// just for TEST
#include<iostream>
#include<string>

#include "layer.h"

class Net {
    std::vector<Layer> layers;

public:
    Net() {
        // Create just input output of 2 layers
        layers.resize(2);
    }

    Net(std::vector<int> layerSizes) {
        layers.resize(layerSizes.size() - 1);
        // I think this will now overflow... hmmm....
        /* because... 
        ex. A(layers.size())
            B(layerSizes.size())
        
        B = 6
        A = B - 1 = 5

        loop up to: B - 1 = 5... ok
        */ 
        for(int i = 0; i + 1 < layerSizes.size(); i++) {
            layers[i] = Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    std::vector<float> feedForward(std::vector<float> initialInput) {
        // Just get the maxSize so I don't resize the vector everytime? is that how vector works?
        int maxSize = layers[0].getOutputSize();
        for(int i = 1; i < layers.size(); i++) {
            if(layers[i].getOutputSize() > maxSize) {
                maxSize = layers[i].getOutputSize();
            }
            if(layers[i].getInputSize() > maxSize) {
                maxSize = layers[i].getInputSize();
            }
        }

        
    }



private:


public:    
    void DEBUG(const std::string& LABEL = "NET TEST") {
        std::cout << "\n" << "v^v " << LABEL << " v^v^v^v^v^v^v^v^v^v" << "\n";
        std::cout << "layers: " << layers.size() << "\n";
        for(int i = 0; i < layers.size(); i++) {
            layers[i].DEBUG("LAYER TEST: " + std::to_string(i));
        }
        std::cout << "\n\n";
    }
};