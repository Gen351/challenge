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
        if(layerSizes.size() < 2) {
            throw std::runtime_error("Network must have at least 2 layers (Input -> Output).");
        }

        layers.resize(layerSizes.size() - 1);
        // I think this will now overflow... hmmm....
        /* ex.  A(layers.size())
                B(layerSizes.size())
        B = 6
        A = B-1 = 5
        loop up to: B-1 = 5... ok
        */ 
        for(int i = 0; i + 1 < layerSizes.size(); i++) {
            layers[i] = Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    void train(const int& iter/*epochs*/,
                const std::vector<std::vector<float>>& trainData,
                const std::vector<std::vector<float>>& targetValues, 
                const float& learningRate) 
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("targetValues.size() doesn't match trainData.size()");
        }

        for(int epoch = 0; epoch < iter; epoch++) {
            for(int i = 0; i < trainData.size(); i++) {
                feedForward(trainData[i]);
                backProp(targetValues[i], learningRate);
            }
        }
    }

    void train(const int& iter/*epochs*/,
                const std::vector<std::vector<float>>& trainData,
                const std::vector<std::vector<float>>& targetValues, 
                const float& learningRate,
                const std::string& print_console) 
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("targetValues.size() doesn't match trainData.size()");
        }

        for(int epoch = 0; epoch < iter; epoch++) {
            float sumError = 0.0f;
            for(int i = 0; i < trainData.size(); i++) {
                feedForward(trainData[i]);
                sumError += backPropWithError(targetValues[i], learningRate);
            }

            float error = sumError / (float)trainData.size();
            std::cerr << "Epoch: " << epoch + 1 << "\tScore: " << error << "\n";
        }
    }

    std::vector<float> predict(const std::vector<float>& input) {
        if(input.size() != layers[0].getLastIntput().size()) {
            std::cerr << "INPUT SIZE doesn't match first layer's INPUT SIZE";
            return input;
        }

        return feedForward(input);
    }

    // still const, this is for saving the network
    const std::vector<Layer>& getLayers() const {
        return layers;
    }

private:
    // doesn't need to be fast for now... so copying is okay
    std::vector<float> feedForward(const std::vector<float>& initialInput) {
        std::vector<float> currentOutput = initialInput;
        
        // feedForward...
        for(int i = 0; i < layers.size(); i++) {
            currentOutput = layers[i].calculateOutput(currentOutput);
        }

        return currentOutput;
    }

    void backProp(const std::vector<float>& targetValues, const float& learningRate) {
        // calculate the initial error
        layers[layers.size() - 1].calculateOutputGradient(targetValues);

        // Create the gradients
        for(int i = layers.size() - 2/*This is safe because there are always >2 layers*/; i >= 0; i--) {
            layers[i].calculateHiddenGradient(layers[i + 1]);
        }
        // Update the weights (I do it backwards because it's called backprop XD)
        for(int i = layers.size() - 1; i >= 0; i--) {
            layers[i].updateWeights(learningRate);
        }
        // one sweep forward, two sweeps backward
    }

    float backPropWithError(const std::vector<float>& targetValues, const float& learningRate) {
        // calculate the initial error
        float error = layers[layers.size() - 1].calculateOutputGradientWithError(targetValues);

        // Create the gradients
        for(int i = layers.size() - 2/*This is safe because there are always >2 layers*/; i >= 0; i--) {
            layers[i].calculateHiddenGradient(layers[i + 1]);
        }
        // Update the weights (I do it backwards because it's called backprop XD)
        for(int i = layers.size() - 1; i >= 0; i--) {
            layers[i].updateWeights(learningRate);
        }
        // one sweep forward, two sweeps backward

        return error;
    }

public:    
    void DEBUG(const std::string& LABEL = "NET TEST") {
        std::cout << "\n" << "v^v " << LABEL << " v^v^v^v^v^v^v^v^v^v" << "\n";
        std::cout << "layers: " << layers.size() << "\n";
        for(int i = 0; i < layers.size(); i++) {
            layers[i].DEBUG("LAYER TEST: " + std::to_string(i));
        }

        std::cout << "Output TEST: \n";
        layers[0].initRandomInput(1, 255);
        layers[0].initRandomBias();
        for(const auto& outputs : feedForward(layers[0].getLastIntput())) {
            std::cout << "~O- " << outputs << "\n";
        }

        std::cout << "\n\n";
    }
};