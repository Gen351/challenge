#pragma once

#include<vector>
#include<cmath>
#include<random>
#include<stdexcept>

// just for TEST
#include<iostream>

class Layer {
    std::vector<std::vector<float>> weight;
    std::vector<float> input;
    std::vector<float> bias;
    std::vector<float> output;

public:
    Layer() {
        weight.resize(1, std::vector<float>(1, initWeight()));
        input.resize(1, 0);
        bias.resize(1, 0);
        output.resize(1, 0);
    }

    Layer(int inputSize, int outputSize) {
        weight.resize(outputSize, std::vector<float>(inputSize, 0));
        for(int i = 0; i < outputSize; i++) {
            for(int j = 0; j < inputSize; j++) {
                weight[i][j] = initWeight();
            }
        }
        input.resize(inputSize, 0);
        bias.resize(outputSize, 0);
        output.resize(outputSize, 0);
    }

    std::vector<float> computeOutput(std::vector<float>& newInput) {
        if(newInput.size() != input.size()) {
            throw std::runtime_error("== Input Sizes DO NOT MATCH ===");
        }
        
        for(int i = 0; i < output.size(); i++) {
            float sum = bias[i];
            for(int j = 0; j < newInput.size(); j++) {
                sum += (newInput[j] * weight[i][j]);
            }
            // need activation? idk where to put activation
            // (1). put sigmoid here?
            output[i] = sigmoid(sum);
        }

        input = newInput;
        return output;
    }

    int getOutputSize() {
        return output.size();
    }
    int getInputSize() {
        return input.size();
    }

private:

    float sigmoid(float x) {
        return (1.0f / (1.0f + std::exp(-x)));
    }

    float initWeight() {
        return ((float)rand()/(float)RAND_MAX)
                - ((float)rand()/(float)RAND_MAX);
    }




public:
    void DEBUG(const std::string& LABEL = "LAYER TEST") {
        std::cout << "\n" << "__ " << LABEL << " ________________________" << "\n";
        std::cout << " weight[" << weight.size() << "]" 
            << "[" << weight[0].size() << "]" 
            << "\t\t| EX: " << weight[0][0] << "\n";
        std::cout << " input: " << input.size() 
            << "\t\t| EX: " << input[0] << "\n";
        std::cout << " output: " << output.size() 
            << "\t\t| EX: " << output[0] << "\n";
        std::cout << " bias: " << bias.size() 
            << "\t\t| EX: " << bias[0] << "\n";
        std::cout << "-----------------------------------------\n";
    }
};