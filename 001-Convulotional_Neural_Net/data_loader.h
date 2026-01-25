#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <utility> 
#include <cmath>     // for std::floor
#include <algorithm> // for std::max

// Include your core matrix and model headers
#include "./cnn/matrix_op/matrix.hpp"
#include "./cnn/sequential.h"
#include "./cnn/tensor.h"

namespace DataLoader {

    // A container for our loaded data
    struct DataSet {
        std::vector<Matrix<float>> inputs;  // 28x28 images
        std::vector<Matrix<float>> targets; // 1x10 one-hot vectors
    };

    // Helper: Map Fashion-MNIST indices to text
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
                // Clamp and convert to 0-255
                float val = img(r, c);
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                int intensity = static_cast<int>(val * 255.0f);
                
                // ANSI Grayscale Background + Two Spaces
                printf("\x1b[48;2;%d;%d;%dm  ", intensity, intensity, intensity);
            }
            std::cout << "\x1b[0m\n"; // Reset at end of row
        }
        std::cout << "\x1b[0m---------------------\n";
    }

    // --- 1. LOAD FUNCTION ---
    inline DataSet loadFashionMNIST(const std::string& path, int limit = -1) {
        std::cout << "Loading dataset: " << path << "..." << std::endl;
        
        DataSet data;
        std::ifstream file(path);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + path);
        }

        std::string line;
        // Skip header if present
        std::getline(file, line); 

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
            data.inputs.push_back(img);
            
            count++;
            if (count % 1000 == 0) {
                std::cout << "\rLoaded " << count << " samples..." << std::flush;
            }
        }
        std::cout << "\rDone! Loaded " << count << " samples.\n" << std::endl;
        return data;
    }

    // --- 2. TRAIN WRAPPER ---
    inline void train(Sequential& model, DataSet& trainData, int epochs, float learningRate) {
        std::cout << "--- Starting Training on " << trainData.inputs.size() << " samples ---" << std::endl;
        model.train(epochs, trainData.inputs, trainData.targets, learningRate);
    }

    // --- 3. EVALUATE WRAPPER ---
    // Added 'visualizeCount' to limit how many images get drawn (printing 100 images is a lot of scrolling!)
    inline void evaluate(Sequential& model, DataSet& testData, int visualizeCount = 5) {
        std::cout << "--- Starting Evaluation ---" << std::endl;
        
        int correct = 0;
        int total = testData.inputs.size();

        for (int i = 0; i < total; i++) {
            // Forward Pass
            Tensor input;
            input.addFeatureMap(testData.inputs[i]);
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

            // Check correctness
            bool isCorrect = (predictedClass == actualClass);
            if (isCorrect) correct++;

            // VISUALIZATION LOGIC
            if (i < visualizeCount) {
                std::string status = isCorrect ? "[CORRECT]" : "[WRONG]";
                std::string header = status + 
                                     " Pred: " + getLabelName(predictedClass) + 
                                     " | Act: " + getLabelName(actualClass);
                
                visualizeMatrix(testData.inputs[i], header);
            }

            if (i % 10 == 0 || i == total - 1) {
                 std::cout << "\rTesting: " << i+1 << "/" << total 
                           << " | Accuracy: " << (float)correct / (i+1) * 100.0f << "%" << std::flush;
            }
        }

        std::cout << "\n\n=== FINAL RESULTS ===" << std::endl;
        std::cout << "Accuracy: " << ((float)correct / total) * 100.0f << "% (" 
                  << correct << "/" << total << ")" << std::endl;
    }
}