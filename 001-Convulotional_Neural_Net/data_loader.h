#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>     // for std::floor
#include <algorithm> // for std::max
#include <cstdlib>   // for system()

#include <random>

// Include your core matrix and model headers
#include "./cnn/matrix_op/matrix.hpp"
#include "./cnn/sequential.h"
#include "./cnn/tensor.h"

// Include Layer implementations for the Factory
#include "./cnn/conv_layer.h"
#include "./cnn/dense_layer.h"
#include "./cnn/pooling_layer.h"
#include "./cnn/activation_layer.h"

namespace DataLoader {

    inline std::vector<int> getRandomIndexes(int max, int count=1, int min=0) {
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

    // --- UPDATED DATASET STRUCT ---
    // Inputs are now Tensors to support both grayscale (1 channel) and RGB (3 channels)
    struct DataSet {
        std::vector<Tensor> inputs;         // <--- CHANGED from Matrix<float>
        std::vector<Matrix<float>> targets; // 1x10 one-hot vectors
    };

    inline std::string getLabelName(int index) {
        const std::string names[] = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
        if (index >= 0 && index < 10) return names[index];
        return "Unknown";
    }

    // --- VISUALIZATION HELPER ---
    inline void visualizeMatrix(const Matrix<float>& img, const std::string& label) {
        std::cout << "\n--- " << label << " ---\n";
        for (size_t r = 0; r < img.rows(); r++) {
            for (size_t c = 0; c < img.cols(); c++) {
                float val = img(r, c);
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                int intensity = static_cast<int>(val * 255.0f);
                
                // ANSI Grayscale Background
                printf("\x1b[48;2;%d;%d;%dm  ", intensity, intensity, intensity);
            }
            std::cout << "\x1b[0m\n"; 
        }
        std::cout << "\x1b[0m---------------------\n";
    }

    // --- 1. LOAD FUNCTION (UPDATED) ---
    inline DataSet loadFashionMNIST(const std::string& path, int limit = -1) {
        std::cout << "Loading dataset: " << path << "..." << std::endl;
        
        DataSet data;
        std::ifstream file(path);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + path);
        }

        std::string line;
        std::getline(file, line); // Skip header

        int count = 0;
        while (std::getline(file, line)) {
            if (limit > 0 && count >= limit) break;

            std::stringstream ss(line);
            std::string token;

            // 1. Get Label
            if (!std::getline(ss, token, ',')) break; 
            int label = std::stoi(token);

            // One-Hot Target
            Matrix<float> target(1, 10, 0.0f);
            target(0, label) = 1.0f;
            data.targets.push_back(target);

            // 2. Get Pixels
            Matrix<float> img(28, 28);
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    if (std::getline(ss, token, ',')) {
                        float pixelVal = std::stof(token);
                        img(r, c) = pixelVal / 255.0f; 
                    }
                }
            }

            // --- CRITICAL FIX: Wrap Matrix in Tensor ---
            Tensor t;
            t.addFeatureMap(img);
            data.inputs.push_back(t);
            // ------------------------------------------
            
            count++;
            if (count % 1000 == 0) {
                std::cout << "\rLoaded " << count << " samples..." << std::flush;
            }
        }
        std::cout << "\rDone! Loaded " << count << " samples.\n" << std::endl;
        return data;
    }

    // --- 2. TRAIN WRAPPER (UPDATED) ---
    inline void train(Sequential& model, DataSet& trainData, int epochs, float learningRate, bool validate=false, bool smart=false) {
        std::cout << "--- Starting Training on " << trainData.inputs.size() << " samples ---" << std::endl;
        
        if(smart) {
            // Now passing vector<Tensor> directly
            model.trainSmart(epochs, trainData.inputs, trainData.targets, learningRate, true);
        } else {
            // Assuming standard train() also accepts Tensors now, 
            // OR you might need to update Sequential::train signature as well.
            // model.train(epochs, trainData.inputs, trainData.targets, learningRate);
            std::cout << "Please use smart=true for Tensor training in this version." << std::endl;
        }
    }

    // --- 3. EVALUATE WRAPPER (UPDATED) ---
    inline void evaluate(Sequential& model, DataSet& testData, int visualizeCount = 5, bool showIncorrect=false) {
        std::cout << "--- Starting Evaluation ---" << std::endl;

        int correct = 0;
        int total = testData.inputs.size();

        std::vector<int> random = getRandomIndexes(total, visualizeCount, 0);
        int visualizeCounter = 0;

        for (int i = 0; i < total; i++) {
            // FIX: Input is already a Tensor
            Tensor input = testData.inputs[i]; 
            
            Tensor output = model.predict(input);

            // Get Predicted Class
            int predictedClass = 0;
            float maxProb = -1.0f;
            const std::vector<float>& outData = output.featureMaps[0].data;
            
            for(size_t j = 0; j < outData.size(); j++) {
                if(outData[j] > maxProb) {
                    maxProb = outData[j];
                    predictedClass = j;
                }
            }

            // Get Actual Class
            int actualClass = 0;
            const std::vector<float>& targetData = testData.targets[i].data;
            for(size_t j = 0; j < targetData.size(); j++) {
                if(targetData[j] > 0.9f) {
                    actualClass = j;
                    break;
                }
            }

            bool isCorrect = (predictedClass == actualClass);
            if (isCorrect) correct++;
            
            if (showIncorrect) {
                if (!isCorrect) {
                    std::cout << "[" << i+1 << "/" << total << "]"
                              << "|[" << (float)correct / (i+1) * 100.0f << "](%)|";
                    std::string header = "[X]Pred:" + getLabelName(predictedClass) + "|Act:" + getLabelName(actualClass);
                    
                    if(visualizeCounter < visualizeCount) {
                        // FIX: Visualize the first feature map (Grayscale image)
                        visualizeMatrix(testData.inputs[i].featureMaps[0], header);
                        visualizeCounter++;
                    } else {
                        std::cout << header;
                    }
                    std::cout << "\n";
                }
            } else {
                // Normal mode
                if (!random.empty() && visualizeCounter < random.size() && i == random[visualizeCounter]) {
                    std::string status = isCorrect ? "[CORRECT]" : "[WRONG]";
                    std::string header = status + " Pred: " + getLabelName(predictedClass) + " | Act: " + getLabelName(actualClass);
                    
                    // FIX: Visualize featureMaps[0]
                    visualizeMatrix(testData.inputs[i].featureMaps[0], header);
                    visualizeCounter++;
                }
            }

            if(!showIncorrect) {
                if (i % 10 == 0 || i == total - 1) {
                    std::cout << "\rTesting: " << i+1 << "/" << total 
                              << " | Accuracy: " << (float)correct / (i+1) * 100.0f << "%" << std::flush;
                }   
            }
        }

        std::cout << "\n\n=== FINAL RESULTS ===" << std::endl;
        std::cout << "Accuracy: " << ((float)correct / total) * 100.0f << "% (" 
                  << correct << "/" << total << ")" << std::endl;
    }

    namespace ModelIO {
        
        const std::string DIR = "./trained_cnns/";

        inline void saveModel(const Sequential& model, const std::string& filename) {
            #ifdef _WIN32
                system(("if not exist \"" + DIR + "\" mkdir \"" + DIR + "\"").c_str());
            #else
                system(("mkdir -p " + DIR).c_str());
            #endif

            std::string fullPath = DIR + filename + ".cnn";
            std::ofstream file(fullPath);
            
            if (!file.is_open()) {
                std::cerr << "Error: Could not open " << fullPath << "\n";
                return;
            }

            const auto& layers = model.getLayers(); 
            file << layers.size() << "\n";

            for (const auto& layer : layers) {
                file << layer->getType() << " "; 
                layer->save(file);
            }

            std::cout << "Model saved to " << fullPath << std::endl;
            file.close();
        }

        inline void loadModel(Sequential& model, const std::string& filename) {
            std::string fullPath = DIR + filename + ".cnn";
            std::ifstream file(fullPath);

            if (!file.is_open()) {
                throw std::runtime_error("Could not open model file: " + fullPath);
            }

            int numLayers;
            file >> numLayers;

            std::cout << "Loading " << numLayers << " layers from " << filename << "...\n";

            for (int i = 0; i < numLayers; i++) {
                std::string type;
                file >> type;

                Layer* newLayer = nullptr;

                if (type == "CONV") {
                    size_t f, d, k;
                    file >> f >> d >> k;
                    newLayer = new ConvLayer(f, d, k);
                    newLayer->load(file);
                }
                else if (type == "DENSE") {
                    size_t in, out;
                    file >> in >> out;
                    newLayer = new DenseLayer(in, out);
                    newLayer->load(file);
                }
                else if (type == "POOL") {
                    int t; size_t s;
                    file >> t >> s;
                    newLayer = new PoolingLayer((PoolType)t, s);
                    newLayer->load(file);
                }
                else if (type == "ACT") {
                    int t;
                    file >> t;
                    newLayer = new ActivationLayer((ActivationType)t);
                    newLayer->load(file);
                }
                else {
                    std::cerr << "Unknown layer type: " << type << std::endl;
                }

                if (newLayer) {
                    model.add(newLayer);
                }
            }
            std::cout << "Model loaded successfully.\n";
            file.close();
        }
    }
}