#pragma once

#include "abstract_layer.h"

class DenseLayer : public Layer {

    Matrix<float> weights;
    std::vector<float> bias;

    Matrix<float> gradientWeights;
    std::vector<float> gradientBias;

    Tensor lastInput;

public:

    DenseLayer(size_t inputSize=1, size_t outputSize=1)
        : weights(inputSize, outputSize)
        , bias(outputSize)
        , gradientWeights(inputSize, outputSize)
        , gradientBias(outputSize)
        , velocityWeights(inputSize, outputSize)
        , velocityBias(outputSize, 0.0f)
    {
        const float scale = std::sqrt(1.0f / inputSize);
        for(float& weight : weights) weight = MatrixOp::initRandFloat() * scale;
    }

    Tensor forward(const Tensor& input) {
        if(training) {
            lastInput = input;
        }
        // flatten
        Matrix<float> flatMatrix(1, bias.size());

        for(size_t i = 0; i < bias.size(); i++) {
            float sum = 0;

            size_t weightRowIndex = 0;
            for(size_t j = 0; j < input.featureMaps.size(); j++) {
                for(size_t k = 0; k < input.featureMaps[j].data.size(); k++) {
                    sum += input.featureMaps[j].data[k] * weights(weightRowIndex, i);
                    weightRowIndex++;
                }
            }
            sum += bias[i];
            flatMatrix(0, i) = sum;
        }

        Tensor output;
        output.addFeatureMap(flatMatrix);
        
        return output;
    }

    Tensor backward(const Tensor& gradientOutput) override {
        // gradientOutput is 1 x OutputSize
        // We need to calculate gradients for: Bias, Weights, and Input

        // A. GRADIENT BIAS
        // For batch size 1, the gradient of the bias is just the output error itself
        for(size_t i = 0; i < gradientBias.size(); i++) {
            gradientBias[i] += gradientOutput.featureMaps[0].data[i];
        }

        // B. GRADIENT WEIGHTS
        // Formula: dWeights = Input_Transposed * Output_Error
        // Since we didn't flatten input into a matrix in forward, we replicate that loop structure here.
        
        size_t inputIndex = 0;
        // Loop over original input structure (Rows of weights)
        for(size_t j = 0; j < lastInput.featureMaps.size(); j++) {
            for(size_t k = 0; k < lastInput.featureMaps[j].data.size(); k++) {
                float inputVal = lastInput.featureMaps[j].data[k];

                // Loop over outputs (Cols of weights)
                for(size_t i = 0; i < gradientOutput.featureMaps[0].data.size(); i++) {
                    float errorVal = gradientOutput.featureMaps[0].data[i];
                    
                    // The weight connecting input[inputIndex] to output[i] contributed 
                    // 'inputVal' amount to the error 'errorVal'
                    gradientWeights(inputIndex, i) += inputVal * errorVal;
                }
                inputIndex++;
            }
        }

        // C. GRADIENT INPUT (The Error to pass back to previous layer)
        // Formula: dInput = Output_Error * Weights_Transposed
        Tensor gradientInput;

        inputIndex = 0;
        // We must reconstruct the EXACT shape of lastInput
        for(size_t j = 0; j < lastInput.featureMaps.size(); j++) {
            
            // Create a matrix matching the size of the feature map from the previous layer
            Matrix<float> channelGrads(lastInput.featureMaps[j].rows(), lastInput.featureMaps[j].cols());

            for(size_t k = 0; k < lastInput.featureMaps[j].data.size(); k++) {
                float errorSum = 0;
                
                // Sum the error contributions from all output nodes
                for(size_t i = 0; i < gradientOutput.featureMaps[0].data.size(); i++) {
                    // weight(inputIndex, i) connects this pixel to output node i
                    errorSum += gradientOutput.featureMaps[0].data[i] * weights(inputIndex, i);
                }
                
                channelGrads.data[k] = errorSum;
                inputIndex++;
            }
            gradientInput.addFeatureMap(channelGrads);
        }

        return gradientInput;
    }

    // by Gemini 3.0 Pro
    void update(float learningRate) override {
        const float momentum = 0.9f;

        // Update Weights with Momentum
        for(size_t i = 0; i < weights.rows(); i++) {
            for(size_t j = 0; j < weights.cols(); j++) {
                // 1. Calculate new velocity
                velocityWeights(i, j) = (momentum * velocityWeights(i, j)) + (learningRate * gradientWeights(i, j));
                
                // 2. Apply velocity to weights
                weights(i, j) -= velocityWeights(i, j);
                
                // 3. Clear gradient for the next pass
                gradientWeights(i, j) = 0.0f; 
            }
        }

        // Update Biases with Momentum
        for(size_t i = 0; i < bias.size(); i++) {
            velocityBias[i] = (momentum * velocityBias[i]) + (learningRate * gradientBias[i]);
            bias[i] -= velocityBias[i];
            gradientBias[i] = 0.0f;
        }
    }

    std::string getType() const override { return "DENSE"; }

    void save(std::ofstream& file) const override {
        // Save Config
        file << weights.rows() << " " << weights.cols() << "\n";

        // Save Weights
        for(float val : weights.data) {
            file << val << " ";
        }

        // Save Biases
        for(float val : bias) {
            file << val << " ";
        }
        file << "\n";
    }

    void load(std::ifstream& file) override {
        // Weights
        for(float& val : weights.data) {
            file >> val;
        }
        // Biases
        for(float& val : bias) {
            file >> val;
        }
    }

    Matrix<float> velocityWeights;
    std::vector<float> velocityBias;
};