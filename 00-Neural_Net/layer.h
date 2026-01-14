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

    std::vector<float> gamma;

public:
    Layer() : weight(1, std::vector<float>(1, initWeight())),
            input(1, 0.0f),
            bias(1, 0.0f),
            output(1, 0.0f),
            gamma(1, 0.0f)
    {}

    Layer(int inputSize, int outputSize) : 
            weight(outputSize, std::vector<float>(inputSize, 0.0f)),
            input(inputSize, 0.0f),
            bias(outputSize, 0.0f),
            output(outputSize, 0.0f),
            gamma(outputSize, 0.0f) 
    {
        for(int i = 0; i < outputSize; i++) {
            for(int j = 0; j < inputSize; j++) {
                weight[i][j] = initWeight();
            }
        }
    }

    Layer(const Layer& other)
        : weight(other.weight),
        input(other.input),
        bias(other.bias),
        output(other.output),
        gamma(other.gamma)
    {}

    // not needed, the compiler is going to do it for me...?
    Layer& operator=(const Layer&) = default;
    Layer(Layer&&) = default;

    const std::vector<float>& calculateOutput(const std::vector<float>& newInput) {
        if(newInput.size() != input.size()) {
            throw std::runtime_error("== Input Sizes DO NOT MATCH ===");
        }
        
        for(int i = 0; i < output.size(); i++) {
            float sum = bias[i];
            for(int j = 0; j < newInput.size(); j++) {
                sum += newInput[j] * weight[i][j];
            }
            // need activation? idk where to put activation
            // (1). put sigmoid here?
            output[i] = sigmoid(sum);
        }

        input = newInput;
        return output;
    }

    void calculateOutputGradient(const std::vector<float>& targetValues) {
        if(targetValues.size() != output.size()) {
            throw std::runtime_error("targetValues.size() do not match output.size()");
        }
        if(gamma.size() != output.size()) {
            throw std::runtime_error("gamma.size() do not match output.size()");
        }

        for(int i = 0; i < targetValues.size(); i++) {
            float difference = targetValues[i] - output[i];
            gamma[i] = difference * sigmoidDerivative(output[i]);
        }
    }

    void calculateHiddenGradient(const Layer& nextLayer) {
        for(int i = 0; i < gamma.size(); i++) {
            float sum = 0.0f;
            for(int j = 0; j < nextLayer.gamma.size(); j++) {
                sum += nextLayer.gamma[j] * nextLayer.weight[j][i];
            }
            gamma[i] = sum * sigmoidDerivative(output[i]); 
        }
    }

    void updateWeights(const float& learningRate) {
        for(int i = 0; i < weight.size(); i++) {
            for(int j = 0; j < weight[i].size(); j++) {
                weight[i][j] = weight[i][j] + (learningRate * gamma[i] * input[j]);
            }
            bias[i] = bias[i] + (learningRate * gamma[i]);
        }
    }

    void getError(const std::vector<float>& targetValues, std::vector<float>& error) {
        if(targetValues.size() != output.size()) {
            throw std::runtime_error("targetValues.size() do not match output.size()");
        } else if(gamma.size() != output.size()) {
            throw std::runtime_error("gamma.size() do not match output.size()");
        } else if(error.size() != output.size()) {
            throw std::runtime_error("gamma.size() do not match output.size()");
        }

        for(int i = 0; i < targetValues.size(); i++) {
            float difference = targetValues[i] - output[i];
            error[i] = difference * difference;
        }
    }
    float calculateOutputGradientWithError(const std::vector<float>& targetValues) {
        if(targetValues.size() != output.size()) {
            throw std::runtime_error("targetValues.size() do not match output.size()");
        }
        if(gamma.size() != output.size()) {
            throw std::runtime_error("gamma.size() do not match output.size()");
        }

        // get the error score
        std::vector<float> error(output.size(), 0.0f);
        getError(targetValues, error);
        float sumError = 0.0f;
        for(const auto& err : error) {
            sumError += err;
        }
        
        for(int i = 0; i < targetValues.size(); i++) {
            float difference = targetValues[i] - output[i];
            gamma[i] = difference * sigmoidDerivative(output[i]);
        }

        return sumError / (float)output.size();
    }

    void setBias(std::vector<float>& newBias) {
        bias = newBias;
    }
    void setWeight(std::vector<std::vector<float>>& newWeight) {
        weight = newWeight;
    }

    /* !! SCARY !!, const, but scary */
    const std::vector<float>& getLastOutput() const {
        return output;
    }
    const std::vector<float>& getLastInput() const {
        return input;
    }
    const std::vector<float>& getGamma() const {
        return gamma;
    }
    const std::vector<float>& getBias() const {
        return bias;
    }
    const std::vector<std::vector<float>>& getWeight() const {
        return weight;
    }


private:

    float sigmoid(float x) {
        return (1.0f / (1.0f + std::exp(-x)));
    }

    float sigmoidDerivative(float val) {
        return (val * (1.0f - val));
    }

    float initWeight() {
        return ((float)rand()/(float)RAND_MAX)
                - ((float)rand()/(float)RAND_MAX);
    }


public:
    void clearInput() {
        for(float& inputs : input) {
            inputs = 0.0f;
        }
    }
    void clearBias() {
        for(float& biases : bias) {
            biases = 0.0f;
        }
    }

    // for testing...
    void initRandomInput(int min = 1, int max = 255) {
        if(min >= max) {
            return;
        }

        for(float& inputs : input) {
            inputs =  (min + (rand() % max));
        }
    }
    void initRandomBias() {
        for(float& biases : bias) {
            biases =  initWeight();
        }
    }

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