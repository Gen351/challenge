#include <iostream>
#include <ctime>
#include <cctype>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip> // Added for nicer printing

// 1. Core Headers
#include "../../cnn/sequential.h"
#include "../../cnn/tensor.h"
#include "../../data_loader.h" 
#include "../../image_loader.h" 

#include "../../cnn/visualize.h"

// --- CONFIGURATION ---
const std::string loadModelName = "cat_dog_breedsRGB4";
const std::string csvPath = "./../cnn-data/catdog/breeds_shuffled_test.csv";

const std::vector<std::string> classes = {"yorkshire_terrier", "english_cocker_spaniel", "samoyed", 
                            "British_Shorthair", "Bengal", "Birman", 
                            "english_setter", "american_pit_bull_terrier", "Bombay", 
                            "chihuahua", "Abyssinian", "havanese", 
                            "Egyptian_Mau", "american_bulldog", "pug", 
                            "beagle", "keeshond", "Sphynx", 
                            "japanese_chin", "Persian", "Maine_Coon", 
                            "miniature_pinscher", "newfoundland", "pomeranian", 
                            "great_pyrenees", "Ragdoll", "german_shorthaired", 
                            "boxer", "leonberger", "Russian_Blue", 
                            "shiba_inu", "saint_bernard", "scottish_terrier", 
                            "Siamese", "staffordshire_bull_terrier", "basset_hound", 
                            "wheaten_terrier"};

// Image Settings
const int IMG_SIZE = 100;
const int CHANNELS = 3;
const int CLASSES = classes.size();
// ---------------------

std::string getClassName(int index) {
    if(index >= 0 && index < classes.size()) return classes[index];
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
            std::cout << "Pred: " << getClassName(predictedClass) << " (" << (int)confidence << "%)";
            std::cout << " | Actual: " << getClassName(actualClass) << std::endl;
            
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
        std::cout << "  1. Test single image:  ./catdog_breeds_test <path_to_image> <show_filters(true | false)>" << std::endl;
        std::cout << "  2. Test dataset:       ./catdog_breeds_test <count(-1 for all)>" << std::endl;
        return -1;
    }

    Sequential model;

    // 1. Load Model
    if (!loadModelName.empty()) {
        std::cout << "Loading model: " << loadModelName << "..." << std::endl;
        try {
            DataLoader::ModelIO::loadModel(model, loadModelName, "../../trained_cnns/catdog/");
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return 1;
        }
    }

    try {
        // --- MODE 1: Single Image Prediction ---
        if(argc > 2) {
            std::string path = argv[1];
            std::cout << "Loading single image: " << path << "..." << std::endl;
            
            // Load and Resize to 64x64x3
            Tensor myImage = ImageLoader::loadImage(path, 100, 100, 3);
            
            // Predict
            Tensor output;
            if(argc == 3 && std::string(argv[2]) == "true" || std::string(argv[2]) == "1") {
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
            std::string predictedClass = getClassName(predictedIndex);
            std::cout << "PREDICTION: " << predictedClass << " [ " << ((isupper(predictedClass[0])) ? "Cat" : "Dog") << " ]" << std::endl;
            std::cout << "CONFIDENCE: " << std::fixed << std::setprecision(3) << confidence << "%" << std::endl;
            
            // Print all probabilities for debugging
            std::cout << "Distribution: [ ";
            for(int i = 0; i < classes.size(); i++) {
                std::cout << getClassName(i) << ": " << (int)(preds[i]*100) << "%\n"; 
            }
            std::cout << "]" << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            Visualize::RGB(myImage);
        }
        // --- MODE 2: Dataset Evaluation ---
        else {
            int visCount = std::stoi(argv[1]);

            auto testData = ImageLoader::loadFromCSV(csvPath, IMG_SIZE, IMG_SIZE, CHANNELS, CLASSES);
            evaluate(model, testData, visCount, false);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
