#pragma once

#include<iostream>

#include<vector>
#include<memory> 

#include "conv_layer.h"
#include "dense_layer.h"
#include "pooling_layer.h"
#include "activation_layer.h"

class Sequential {
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(Layer* layer) {
        layers.push_back(std::unique_ptr<Layer>(layer));
    }


    Tensor predict(const Tensor& input) {        
        for(auto& layer : layers) layer->predict();
        
        Tensor current = input;
        for(auto& layer : layers) {
            current = layer->forward(current);
        }

        return current;
    }


    void train(const size_t epochs,
                const std::vector<Matrix<float>>& trainData,
                const std::vector<Matrix<float>>& targetValues,
                float learningRate)
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("Train_data size != Target_values size");
        }

        // Set all layers to training mode
        for(auto& layer : layers) layer->train();

        size_t half = epochs / 2;
        for(size_t epoch = 0; epoch < epochs; epoch++) {
            
            float totalLoss = 0;

            // Iterate over every sample in the dataset
            for(size_t i = 0; i < trainData.size(); i++) {

                // A. CONVERT DATA TO TENSOR
                // The layers expect Tensors (which hold feature maps). 
                // We assume trainData[i] is a single channel image.
                Tensor input;
                input.addFeatureMap(trainData[i]);

                Tensor target;
                target.addFeatureMap(targetValues[i]);

                // B. FORWARD PASS
                Tensor output = input;
                for(auto& layer : layers) {
                    output = layer->forward(output);
                }

                // C. CALCULATE LOSS GRADIENT (MSE)
                // We need to calculate the error at the very end to start backpropagation.
                // MSE Derivative: 2 * (Output - Target)
                Tensor lossGradient = output; // Copy dimensions
                
                // Assuming output is 1xN (Dense Layer output)
                float sampleLoss = 0;
                std::vector<float>& outData = output.featureMaps[0].data;
                std::vector<float>& targetData = target.featureMaps[0].data;
                std::vector<float>& gradData = lossGradient.featureMaps[0].data;

                for(size_t j = 0; j < outData.size(); j++) {
                    float error = outData[j] - targetData[j];
                    sampleLoss += error * error;
                    
                    // The gradient we send into the last layer's backward()
                    gradData[j] = 2.0f * error; 
                }
                totalLoss += sampleLoss;

                // D. BACKWARD PASS
                // Loop in reverse order
                Tensor currentGradient = lossGradient;
                for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    currentGradient = (*it)->backward(currentGradient);
                }

                // E. UPDATE WEIGHTS
                for(auto& layer : layers) {
                    layer->update(learningRate);
                }
            }

            // Optional: Print average loss per epoch
            printf("Epoch: %3d | Loss: %.9f | Lrate: %.8f\n", epoch + 1, (totalLoss / trainData.size()), learningRate);
            size_t remaining = epochs - epoch;
            if(remaining <= half && half > 0) {
                learningRate /= 15;
                half = remaining / 2;
            }
        }
    }


    const std::vector<std::unique_ptr<Layer>>& getLayers() const {
        return layers;
    }

private:

};