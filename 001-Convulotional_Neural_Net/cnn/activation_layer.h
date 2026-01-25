#pragma once

#include "abstract_layer.h"

enum class ActivationType {
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    SoftMax
};

typedef float (*ActivationFunc)(const float&);

class ActivationLayer : public Layer {

    ActivationType type;
    ActivationFunc activation = nullptr;
    ActivationFunc derivative = nullptr;
    Tensor lastInput;
    
public:

    ActivationLayer(ActivationType initType=ActivationType::Linear) 
        : type(initType) {
        if (initType == ActivationType::Linear) {
            activation = &linear;
            derivative = &linearDerivative;
        } else if (initType == ActivationType::ReLU) {
            activation = &ReLU;
            derivative = &ReLUDerivative;
        } else if (initType == ActivationType::LeakyReLU) {
            activation = &leakyReLU;
            derivative = &leakyReLUDerivative;
        } else if (initType == ActivationType::Sigmoid) {
            activation = &sigmoid;
            derivative = &sigmoidDerivative;
        } else if (initType == ActivationType::Tanh) {
            activation = &tanh;
            derivative = &tanhDerivative;
        }
        // SoftMax doesn't use the simple float pointers
    }
    
    Tensor forward(const Tensor& input) override {
        if(training) {
            lastInput = input;
        }

        if(type == ActivationType::Linear) {
            return input;
        }

        Tensor output = input;

        if(type == ActivationType::SoftMax) {
            softmax(output);
        } else {
            for(auto& featureMap : output.featureMaps) {
                for(float& data : featureMap) {
                    data = activation(data);
                }
            }
        }

        return output;
    }
    

    Tensor backward(const Tensor& gradientOutput) override {
        // If Linear or SoftMax (when combined with CrossEntropy), we usually pass gradient through.
        // NOTE: True Softmax derivative is a Jacobian matrix, usually handled in the Loss function.
        if(type == ActivationType::Linear || type == ActivationType::SoftMax) {
            return gradientOutput;
        }

        Tensor gradientInput = gradientOutput; // Copy dimensions

        // CHAIN RULE: Gradient = IncomingGradient * Derivative(LastInput)
        for(size_t i = 0; i < gradientInput.featureMaps.size(); i++) {
             // Loop through every pixel/neuron
             for(size_t j = 0; j < gradientInput.featureMaps[i].data.size(); j++) {
                 
                 float inputVal = lastInput.featureMaps[i].data[j];
                 float derivativeVal = derivative(inputVal);
                 
                 // Apply Chain Rule
                 gradientInput.featureMaps[i].data[j] = gradientOutput.featureMaps[i].data[j] * derivativeVal;
             }
        }
        
        return gradientInput;
    }

    virtual void update(float learningRate) override {

    }
    
private:

    static void softmax(Tensor& input) {
        float maxVal = -INFINITY;
        
        for(auto& featureMap : input.featureMaps) {
            for(float& data : featureMap) {
                if(data > maxVal) {
                    maxVal = data;
                }
            }
        }
        
        float sum = 0.0f;
        for(auto& featureMap : input.featureMaps) {
            for(float& data : featureMap) {
                data = std::exp(data - maxVal);
                sum += data;
            }
        }
 
        for(auto& featureMap : input.featureMaps) {
            for(float& data : featureMap) {
                data /= sum;
            }
        }
    }

    static float linear(const float& x) {
        return x;
    }
    static float linearDerivative(const float& x) {
        return 1.0f;
    }

    static float sigmoid(const float& x) {
        return (1.0f / (1.0f + std::exp(-x)));
    }
    static float sigmoidDerivative(const float& x) {
        const float sigmoidD = sigmoid(x);
        return sigmoidD * (1.0f - sigmoidD);
    }

    static float ReLU(const float& val) {
        return val * (val > 0);
    }
    static float ReLUDerivative(const float& val) {
        return val > 0;
    }

    static float leakyReLU(const float& val) {
        return val < 0 ? val * 0.01f : val;
    }
    static float leakyReLUDerivative(const float& val) {
        return val < 0 ? 0.01f : 1.0f;
    }

    static float tanh(const float& x) {
        return std::tanh(x);
    }
    static float tanhDerivative(const float& x) {
        const float tanhx = std::tanh(x);
        
        return 1.0f - tanhx * tanhx;
    }
};