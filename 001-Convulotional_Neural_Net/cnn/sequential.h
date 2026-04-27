#pragma once

#include<iostream>

#include<algorithm>
#include<vector>
#include<memory> 

#include "conv_layer.h"
#include "dense_layer.h"
#include "pooling_layer.h"
#include "activation_layer.h"

#include "visualize.h"

class Sequential {
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(Layer* layer) {
        layers.push_back(std::unique_ptr<Layer>(layer));
    }

    Tensor predict(const Tensor& input, bool visualize = false) {
        for (auto& layer : layers) layer->predict();

        Tensor current = input;
        int layerIndex = 0;

        if (visualize) {
            std::cout << "\n--- Input Image ---" << std::endl;
            Visualize::image(current);
        }

        for (auto& layer : layers) {
            current = layer->forward(current);

            if (visualize) {
                std::string layerType = layer->getType();

                if (layerType == "CONV" || layerType == "POOL") {
                    std::cout << "\n--- Layer " << layerIndex << " [" << layerType << "] ---" << std::endl;

                    size_t numMaps = current.featureMaps.size();
                    size_t rows = current.featureMaps[0].rows();
                    size_t cols = current.featureMaps[0].cols();

                    std::vector<float> maxVals(numMaps, 0.001f);
                    for (size_t m = 0; m < numMaps; m++) {
                        for (float val : current.featureMaps[m].data) {
                            if (val > maxVals[m]) maxVals[m] = val;
                        }
                    }

                    std::string buffer;
                    buffer.reserve(numMaps * rows * cols * 35); // Slightly larger buffer for heatmaps
                    char pxBuf[64];

                    size_t mapsPerRow = 16;
                    if (cols >= 128) mapsPerRow = 1;
                    else if (cols >= 64) mapsPerRow = 2;
                    else if (cols >= 32) mapsPerRow = 4;
                    else if (cols >= 16) mapsPerRow = 8;

                    for (size_t chunkStart = 0; chunkStart < numMaps; chunkStart += mapsPerRow) {
                        size_t chunkEnd = std::min(chunkStart + mapsPerRow, numMaps);

                        for (size_t r = 0; r < rows; r++) {
                            for (size_t m = chunkStart; m < chunkEnd; m++) {
                                for (size_t c = 0; c < cols; c++) {
                                    float val = current.featureMaps[m](r, c);
                                    float norm = val / maxVals[m];
                                    if (norm < 0) norm = 0;
                                    if (norm > 1) norm = 1;

                                    // --- HEATMAP COLOR CALCULATION (Jet Approximation) ---
                                    int r_val, g_val, b_val;
                                    
                                    // Intensity 0.0 (Blue) -> 0.5 (Green) -> 1.0 (Red)
                                    if (norm < 0.25f) {
                                        r_val = 0; g_val = static_cast<int>(norm * 4 * 255); b_val = 255;
                                    } else if (norm < 0.5f) {
                                        r_val = 0; g_val = 255; b_val = static_cast<int>((0.5f - norm) * 4 * 255);
                                    } else if (norm < 0.75f) {
                                        r_val = static_cast<int>((norm - 0.5f) * 4 * 255); g_val = 255; b_val = 0;
                                    } else {
                                        r_val = 255; g_val = static_cast<int>((1.0f - norm) * 4 * 255); b_val = 0;
                                    }

                                    int len = snprintf(pxBuf, sizeof(pxBuf), "\x1b[48;2;%d;%d;%dm  ", r_val, g_val, b_val);
                                    buffer.append(pxBuf, len);
                                }
                                buffer.append("\x1b[0m  ");
                            }
                            buffer.append("\x1b[0m\n");
                        }
                        buffer.append("\n\n");
                    }

                    fwrite(buffer.c_str(), 1, buffer.size(), stdout);
                    std::cout << "=========================================\n";
                }
            }
            layerIndex++;
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


    void trainSmart(const size_t epochs,
                std::vector<Tensor>& trainData,  // <--- CHANGED from vector<Matrix>
                std::vector<Matrix<float>>& targetValues,
                float learningRate, bool flip=true, int batchSize=30)
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("Train_data size != Target_values size");
        }

        for(auto& layer : layers) layer->train();

        std::cout << "Training {\n\tEpoch [" << epochs 
                        << "]\n\tLearn Rate [" << learningRate 
                        << "]\n\tBatchSize [" << batchSize 
                        << "]\n\tTrain Data Size [" << trainData.size() << "]\n}\n"; 

        const int MAX_PATIENCE = 10; 
        int patienceCounter = 0;
        float bestLoss = -1.0f;

        for(size_t epoch = 0; epoch < epochs; epoch++) {
            
            float totalLoss = 0;

            for(size_t i = 0; i < trainData.size(); i++) {
                // 1. Input Setup 
                Tensor input = trainData[i];
                
                int toss = rand() % 101;
                // --- AUGMENTATION FOR RGB TENSORS ---
                // 70% chance to modify the input for more robust training....
                if(flip && toss < 70) { // 70%
                    toss = rand() % 2;
                    if(toss == 0) {
                        for(auto& map : input.featureMaps) MatrixOp::flipMatrixRef(map);
                    } else {
                        int dir = rand() % 4;
                        int amount = rand() % 4 + 1;
                        for(auto& map : input.featureMaps) {
                            map = MatrixOp::shift(map, dir, amount, 0.0f);
                        }
                    }
                }

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
                if((i + 1) % batchSize == 0 || i == trainData.size() - 1) {
                    int currentBatchSize = ((i + 1) % batchSize == 0) ? batchSize : (trainData.size() % batchSize);
                    
                    for(auto& layer : layers) layer->update(learningRate / currentBatchSize);
                }
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
        }
    }

    void trainSmartCrossEntropy(const size_t epochs,
                std::vector<Tensor>& trainData,  // <--- CHANGED from vector<Matrix>
                std::vector<Matrix<float>>& targetValues,
                float learningRate, bool flip=true, int batchSize=30)
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("Train_data size != Target_values size");
        }

        for(auto& layer : layers) layer->train();

        std::cout << "Training {\n\tEpoch [" << epochs 
                        << "]\n\tLearn Rate [" << learningRate 
                        << "]\n\tBatchSize [" << batchSize 
                        << "]\n\tTrain Data Size [" << trainData.size() << "]\n}\n"; 

        const int MAX_PATIENCE = 30; 
        int patienceCounter = 0;
        float bestLoss = -1.0f;

        for(size_t epoch = 0; epoch < epochs; epoch++) {
            
            float totalLoss = 0;

            for(size_t i = 0; i < trainData.size(); i++) {
                // 1. Input Setup 
                Tensor input = trainData[i];
                
                int toss = rand() % 101;
                // --- AUGMENTATION FOR RGB TENSORS ---
                // 70% chance to modify the input for more robust training....
                if(flip && toss < 70) { // 70%
                    toss = rand() % 2;
                    if(toss == 0) {
                        for(auto& map : input.featureMaps) MatrixOp::flipMatrixRef(map);
                    } else {
                        int dir = rand() % 4;
                        int amount = rand() % 4 + 1;
                        for(auto& map : input.featureMaps) {
                            map = MatrixOp::shift(map, dir, amount, 0.0f);
                        }
                    }
                }

                Tensor target;
                target.addFeatureMap(targetValues[i]);

                // 2. Forward Pass
                Tensor output = input;
                for(auto& layer : layers) output = layer->forward(output);

                // 3. Loss Calculation (Categorical Cross-Entropy)
                Tensor lossGradient = output; 
                float sampleLoss = 0;
                
                std::vector<float>& outData = output.featureMaps[0].data;
                std::vector<float>& targetData = target.featureMaps[0].data;
                std::vector<float>& gradData = lossGradient.featureMaps[0].data;

                // Epsilon prevents log(0) and division by zero which causes "NaN" (Not a Number) crashes
                const float epsilon = 1e-7f; 

                for(size_t j = 0; j < outData.size(); j++) {
                    // Clip the output so it never hits absolute 0.0 or absolute 1.0
                    float val = std::max(epsilon, std::min(1.0f - epsilon, outData[j]));
                    
                    // Cross-Entropy Loss Calculation: -y * log(a)
                    if (targetData[j] > 0.0f) {
                        sampleLoss += -targetData[j] * std::log(val);
                    }
                    
                    // The Derivative of Cross-Entropy w.r.t the SoftMax output
                    gradData[j] = val - targetData[j]; 
                }
                totalLoss += sampleLoss;

                // 4. Backward Pass
                Tensor currentGradient = lossGradient;
                for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    currentGradient = (*it)->backward(currentGradient);
                }

                // 5. Update Weights
                if((i + 1) % batchSize == 0 || i == trainData.size() - 1) {
                    int currentBatchSize = ((i + 1) % batchSize == 0) ? batchSize : (trainData.size() % batchSize);
                    
                    for(auto& layer : layers) layer->update(learningRate / currentBatchSize);
                }
            }

            float avgLoss = totalLoss / trainData.size();
            printf("Epoch: %3d | Loss: %.9f | Lrate: %.8f\n", epoch + 1, avgLoss, learningRate);

            // --- PATIENCE LOGIC ---
            if (bestLoss < 0 || avgLoss < bestLoss) {
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
        }
    }


    const std::vector<std::unique_ptr<Layer>>& getLayers() const {
        return layers;
    }

private:
};