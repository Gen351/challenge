#pragma once

#include<iostream>

#include<algorithm>
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

        float previousLoss = -1.0f;

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
            float avgLoss = totalLoss / trainData.size();
            printf("Epoch: %3d | Loss: %.9f | Lrate: %.8f\n", epoch + 1, avgLoss, learningRate);

            // DECAY LOGIC:
            if (previousLoss > 0) {
                float improvement = previousLoss - avgLoss;
                
                // If improvement is less than 0.1%, drop learning rate by 10x
                if (improvement < (previousLoss * 0.001f)) {
                    learningRate *= 0.1f;
                    printf("--- Learning rate dropped to %.8f ---\n", learningRate);
                }
            }
            previousLoss = avgLoss;
        }
    }


    // 1. Update arguments to accept Tensor inputs
    void trainSmart(const size_t epochs,
                std::vector<Tensor>& trainData,  // <--- CHANGED from vector<Matrix>
                std::vector<Matrix<float>>& targetValues,
                float learningRate, bool flip=true)
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("Train_data size != Target_values size");
        }

        size_t shiftSize = (size_t)(trainData.size() * 0.7); 
        // We make a copy of TENSORS now, not Matrices
        std::vector<Tensor> trainDataCopy = trainData; 
        std::vector<int> shiftedNFlipped;

        for(auto& layer : layers) layer->train();

        float bestLoss = -1.0f;
        int patienceCounter = 0;
        const int MAX_PATIENCE = 3; 

        for(size_t epoch = 0; epoch < epochs; epoch++) {
            
            // --- AUGMENTATION FOR RGB TENSORS ---
            if(flip) {
                std::vector<int> randomIndexes = getRandomIndexes((int)trainData.size(), (int)shiftSize, 0);            
                shiftedNFlipped = randomIndexes;

                for(int idx : randomIndexes) {
                    int roll = rand() % 2;
                    // Apply augmentation to ALL feature maps (R, G, B)
                    Tensor& t = trainData[idx];
                    
                    if(roll == 0) { 
                        // Flip all channels
                        for(auto& map : t.featureMaps) {
                            MatrixOp::flipMatrixRef(map);
                        }
                    } else {
                        // Shift all channels identically
                        int dir = rand() % 4;
                        int amount = rand() % 2 + 1;
                        for(auto& map : t.featureMaps) {
                            map = MatrixOp::shift(map, dir, amount, 0.0f);
                        }
                    }
                }
            }

            float totalLoss = 0;

            for(size_t i = 0; i < trainData.size(); i++) {
                // 1. Input Setup 
                // Since trainData is already a vector of Tensors, we just copy it
                Tensor input = trainData[i]; 

                Tensor target;
                target.addFeatureMap(targetValues[i]);

                // 2. Forward Pass
                Tensor output = input;
                for(auto& layer : layers) output = layer->forward(output);

                // 3. Loss Calculation
                Tensor lossGradient = output; 
                float sampleLoss = 0;
                
                // Note: Output is likely 1D (Dense Layer), so featureMaps[0] is correct
                std::vector<float>& outData = output.featureMaps[0].data;
                std::vector<float>& targetData = target.featureMaps[0].data;
                std::vector<float>& gradData = lossGradient.featureMaps[0].data;

                for(size_t j = 0; j < outData.size(); j++) {
                    float error = outData[j] - targetData[j];
                    sampleLoss += error * error;
                    gradData[j] = 2.0f * error; 
                }
                totalLoss += sampleLoss;

                // 4. Backward Pass
                Tensor currentGradient = lossGradient;
                for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    currentGradient = (*it)->backward(currentGradient);
                }

                // 5. Update Weights
                for(auto& layer : layers) layer->update(learningRate);
            }

            float avgLoss = totalLoss / trainData.size();
            printf("Epoch: %3d | Loss: %.9f | Lrate: %.8f\n", epoch + 1, avgLoss, learningRate);

            // --- PATIENCE LOGIC ---
            if (bestLoss < 0 || avgLoss < bestLoss * 0.999f) {
                bestLoss = avgLoss;
                patienceCounter = 0; 
            } else {
                patienceCounter++;
                printf("!!! No improvement. Patience: %d/%d\n", patienceCounter, MAX_PATIENCE);
                
                if (patienceCounter >= MAX_PATIENCE) {
                    learningRate *= 0.5f; 
                    printf("--- Learning rate dropped to %.8f ---\n", learningRate);
                    patienceCounter = 0;
                }
            }

            // --- RESTORE ORIGINAL DATA ---
            if(flip) {
                // Restore the specific tensors we modified
                for(int idx : shiftedNFlipped) {
                    trainData[idx] = trainDataCopy[idx]; 
                }
            }
        }
    }


    const std::vector<std::unique_ptr<Layer>>& getLayers() const {
        return layers;
    }

private:

    std::vector<int> getRandomIndexes(int max, int count=1, int min=0) {
        min = min < 0 ? 0 : min;
        max = max < min ? min : max;
        count = count < 1 ? 1 : count;

        std::vector<int> random(count);
        for(int i = 0; i < count; i++) {
            random[i] = rand() % max + min;
        }

        std::sort(random.begin(), random.end());
        return random;
    }


};