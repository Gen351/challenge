#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip> // Added for nicer printing

// 1. Core Headers
#include "cnn/sequential.h"
#include "cnn/tensor.h"
#include "data_loader.h" 
#include "image_loader.h" 

#include "cnn/visualize.h"

// --- CONFIGURATION ---
const std::string loadModelName = "cat_dog_RGB0";
const std::string csvPath = "cnn-data/catdog/CatDog_test.csv";

std::string getCatDogName(int index) {
    const char* names[] = {"Cat", "Dog"};
    if(index >= 0 && index < 2) return names[index];
    return "Unknown";
}




// Helper: Batch Evaluation
void evaluate(Sequential& model, const ImageLoader::Dataset& data, int visCount, bool showIncorrectOnly) {
    int correct = 0;
    int displayed = 0;
    
    std::cout << "\n--- Evaluating Model on " << data.inputs.size() << " images ---" << std::endl;

    for(size_t i = 0; i < data.inputs.size(); i++) {
        Tensor output = model.predict(data.inputs[i]);
        
        std::vector<float>& outPreds = output.featureMaps[0].data;
        auto maxPredIt = std::max_element(outPreds.begin(), outPreds.end());
        int predictedClass = std::distance(outPreds.begin(), maxPredIt);
        float confidence = *maxPredIt * 100.0f;

        const std::vector<float>& outTrue = data.targets[i].data; 
        auto maxTrueIt = std::max_element(outTrue.begin(), outTrue.end());
        int actualClass = std::distance(outTrue.begin(), maxTrueIt);

        bool isCorrect = (predictedClass == actualClass);
        if(isCorrect) correct++;

        if (displayed < visCount) {
            if (showIncorrectOnly && isCorrect) continue;

            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "Sample " << i << ": " << (isCorrect ? "✅ " : "❌ ");
            std::cout << "Pred: " << getCatDogName(predictedClass) << " (" << (int)confidence << "%)";
            std::cout << " | Actual: " << getCatDogName(actualClass) << std::endl;
            
            Visualize::RGB(data.inputs[i]);
            displayed++;
        }
    }

    float accuracy = (float)correct / data.inputs.size() * 100.0f;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Final Accuracy: " << accuracy << "% (" << correct << "/" << data.inputs.size() << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    // Usage Check
    if(argc < 2) {
        std::cout << "Usage Options:" << std::endl;
        std::cout << "  1. Test single image:  ./catdog_test <path_to_image>" << std::endl;
        std::cout << "  2. Test dataset:       ./catdog_test <count> <show_incorrect_only>" << std::endl;
        return -1;
    }

    Sequential model;

    // 1. Load Model
    if (!loadModelName.empty()) {
        std::cout << "Loading model: " << loadModelName << "..." << std::endl;
        try {
            DataLoader::ModelIO::loadModel(model, loadModelName);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return 1;
        }
    }

    try {
        // --- MODE 1: Single Image Prediction ---
        if(argc > 2 && argv[2] != "0" || argv[2] != "1" || argv[2] != "true" || argv[2] != "false") {
            std::string path = argv[1];
            std::cout << "Loading single image: " << path << "..." << std::endl;
            
            // Load and Resize to 64x64x3
            Tensor myImage = ImageLoader::loadImage(path, 64, 64, 3);
            
            // Predict
            Tensor output;
            if(argc == 3 && std::string(argv[2]) == "show_filters") {
                output = model.predict(myImage, true);
            } else {
                output = model.predict(myImage);
            }
            
            // Read Output Vector
            std::vector<float>& preds = output.featureMaps[0].data;
            auto maxIt = std::max_element(preds.begin(), preds.end());
            int predictedIndex = std::distance(preds.begin(), maxIt);
            float confidence = *maxIt * 100.0f;

            // Visualize
            std::cout << "---------------------------------------" << std::endl;
            std::cout << "PREDICTION: " << getCatDogName(predictedIndex) << std::endl;
            std::cout << "CONFIDENCE: " << std::fixed << std::setprecision(2) << confidence << "%" << std::endl;
            
            // Print all probabilities for debugging
            std::cout << "Distribution: [ ";
            for(int i=0; i<2; i++) {
                std::cout << getCatDogName(i) << ":" << (int)(preds[i]*100) << "% "; 
            }
            std::cout << "]" << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            Visualize::RGB(myImage);
        }
        // --- MODE 2: Dataset Evaluation ---
        else {
            int visCount = std::stoi(argv[1]);
            std::string argFlag = argv[2];
            bool showIncorrect = (argFlag == "1" || argFlag == "true");

            auto testData = ImageLoader::loadFromCSV(csvPath, 64, 64, 3, 5);
            evaluate(model, testData, visCount, showIncorrect);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
