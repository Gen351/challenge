#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>

// 1. Include your Core Headers
#include "../../cnn/sequential.h"
#include "../../cnn/tensor.h"
#include "../../cnn/matrix_op/matrix.hpp"

// 2. Include Layer Implementations
#include "../../cnn/conv_layer.h"
#include "../../cnn/pooling_layer.h"
#include "../../cnn/dense_layer.h"
#include "../../cnn/activation_layer.h"

// 3. Include Loaders & Visualization
#include "../../image_loader.h" 
#include "../../data_loader.h"  
#include "../../cnn/visualize.h"

#include <windows.h>
#include <commdlg.h>

// Context
int IMG_SIZE = 64; // Default for the simple Cat vs Dog model
int CHANNELS = 3;
const int CLASSES = 2;
// ---------------------

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
            
            // Replaced RGB with image()
            Visualize::image(data.inputs[i]);
            displayed++;
        }
    }

    float accuracy = (float)correct / data.inputs.size() * 100.0f;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Final Accuracy: " << accuracy << "% (" << correct << "/" << data.inputs.size() << ")" << std::endl;
}

std::string openFile() {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "All\0*.*\0CSV Files\0*.CSV\0"; // Updated filter to include CSV
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn) == TRUE) {
        return std::string(ofn.lpstrFile);
    }
    return "";
}

int main(int argc, char* argv[]) {
    std::srand(std::time(nullptr));

    char op = '0';
    bool exit = false;

    Sequential model;
    bool modelLoaded = false;
    std::string imagePath = "";

    while(!exit) {
        op = '0'; // Reset operation
        std::cout << "\n====================================\n";
        std::cout << "      CAT vs DOG INFERENCE MENU     \n";
        std::cout << "====================================\n";
        while(op > '7' || op < '1') {
            system("cls");
            std::cout << "(1) [Load Model]         " << (modelLoaded ? "(Loaded)" : "(Empty)") << "\n";
            std::cout << "(2) [Load Image]         " << (!imagePath.empty() ? "(Ready)" : "(Empty)") << "\n";
            std::cout << "(3) [Run Single Test]\n";
            std::cout << "(4) [Run Single Test (show filters)]\n";
            std::cout << "(5) [Test Whole Dataset] (CSV)\n";
            std::cout << "(6) [Set Context]\n";
            std::cout << "(7) [Exit]\n -$ ";
            std::cin >> op;
        }
        
        if(op == '7') { exit = true; break; }

        switch(op) {
            case '1': {
                std::cout << "\nOpening File Explorer to select Model...\n";
                std::string fullPath = openFile();
                
                if (!fullPath.empty()) {
                    std::string directory = "";
                    std::string filename = fullPath;
                    const size_t last_slash = fullPath.find_last_of("\\/");
                    if (std::string::npos != last_slash) {
                        directory = fullPath.substr(0, last_slash + 1);
                        filename = fullPath.substr(last_slash + 1);
                        
                        size_t ext_idx = filename.find(".cnn");
                        if (ext_idx != std::string::npos) filename.erase(ext_idx);
                    }

                    try {
                        std::cout << "Loading Model: " << filename << " from " << directory << "...\n";
                        DataLoader::ModelIO::loadModel(model, filename, directory);
                        modelLoaded = true;
                        std::cout << "[SUCCESS] Model Loaded!\n";
                    } catch (const std::exception& e) {
                        std::cerr << "[ERROR] Failed to load model: " << e.what() << "\n";
                    }
                } else {
                    std::cout << "[WARNING] No file selected.\n";
                }
                break;
            }

            case '2': {
                std::cout << "\nOpening File Explorer to select Image...\n";
                imagePath = openFile();
                if (!imagePath.empty()) {
                    std::cout << "[SUCCESS] Selected Image: " << imagePath << "\n";
                } else {
                    std::cout << "[WARNING] No image selected.\n";
                }
                break;
            }

            case '3': {
                if (!modelLoaded) {
                    std::cout << "[ERROR] You must load a model first (Option 1)!\n";
                    std::cin.get();
                    break;
                }
                if (imagePath.empty()) {
                    std::cout << "[ERROR] You must load an image first (Option 2)!\n";
                    std::cin.get();
                    break;
                }

                std::cout << "\n--- Running Inference ---\n";
                try {
                    Tensor myImage = ImageLoader::loadImage(imagePath, IMG_SIZE, IMG_SIZE, CHANNELS);
                    Tensor output = model.predict(myImage);
                    
                    std::vector<float>& preds = output.featureMaps[0].data;
                    auto maxIt = std::max_element(preds.begin(), preds.end());
                    int predictedIndex = std::distance(preds.begin(), maxIt);
                    float confidence = *maxIt * 100.0f;

                    std::cout << "---------------------------------------\n";
                    std::cout << "PREDICTION: " << getCatDogName(predictedIndex) << "\n";
                    std::cout << "CONFIDENCE: " << std::fixed << std::setprecision(2) << confidence << "%\n";
                    
                    std::cout << "Distribution: [ ";
                    for(int i = 0; i < CLASSES; i++) {
                        std::cout << getCatDogName(i) << ":" << (int)(preds[i]*100) << "% "; 
                    }
                    std::cout << "]\n";
                    std::cout << "---------------------------------------\n";

                    Visualize::image(myImage);
                    
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Inference failed: " << e.what() << "\n";
                }

                system("pause");
                break;
            }


            case '4': {
                if (!modelLoaded) {
                    std::cout << "[ERROR] You must load a model first (Option 1)!\n";
                    std::cin.get();
                    break;
                }
                if (imagePath.empty()) {
                    std::cout << "[ERROR] You must load an image first (Option 2)!\n";
                    std::cin.get();
                    break;
                }

                std::cout << "\n--- Running Inference ---\n";
                try {
                    Tensor myImage = ImageLoader::loadImage(imagePath, IMG_SIZE, IMG_SIZE, CHANNELS);
                    Tensor output = model.predict(myImage, true);
                    
                    std::vector<float>& preds = output.featureMaps[0].data;
                    auto maxIt = std::max_element(preds.begin(), preds.end());
                    int predictedIndex = std::distance(preds.begin(), maxIt);
                    float confidence = *maxIt * 100.0f;

                    std::cout << "---------------------------------------\n";
                    std::cout << "PREDICTION: " << getCatDogName(predictedIndex) << "\n";
                    std::cout << "CONFIDENCE: " << std::fixed << std::setprecision(2) << confidence << "%\n";
                    
                    std::cout << "Distribution: [ ";
                    for(int i = 0; i < CLASSES; i++) {
                        std::cout << getCatDogName(i) << ":" << (int)(preds[i]*100) << "% "; 
                    }
                    std::cout << "]\n";
                    std::cout << "---------------------------------------\n";

                    Visualize::image(myImage);
                    
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Inference failed: " << e.what() << "\n";
                }

                system("pause");
                break;
            }

            case '5': {
                if (!modelLoaded) {
                    std::cout << "[ERROR] You must load a model first (Option 1)!\n";
                    break;
                }

                std::cout << "\nOpening File Explorer to select Test CSV...\n";
                std::string csvPath = openFile();
                
                if (!csvPath.empty()) {
                    int visCount = 0;
                    std::cout << "How many samples do you want to visually display? (Enter 0 for none): ";
                    std::cin >> visCount;

                    std::cout << "\n--- Loading Dataset ---\n";
                    try {
                        auto testData = ImageLoader::loadFromCSV(csvPath, IMG_SIZE, IMG_SIZE, CHANNELS, CLASSES);
                        evaluate(model, testData, visCount, false);
                    } catch (const std::exception& e) {
                        std::cerr << "[ERROR] Dataset evaluation failed: " << e.what() << "\n";
                    }
                } else {
                    std::cout << "[WARNING] No CSV selected.\n";
                }
                std::cin.get();
                break;
            }
            
            case '6': {
                std::cout << "\n--- Set Context ---\n";
                std::cout << "Current Image Size: " << IMG_SIZE << "x" << IMG_SIZE << "\n";
                std::cout << "Enter new Image Size (e.g., 64): ";
                std::cin >> IMG_SIZE;
                std::cout << "[SUCCESS] Context Updated to " << IMG_SIZE << "x" << IMG_SIZE << "\n";
                std::cin.get();
                break;
            }
        }
    }
    
    std::cout << "Exiting...\n";
    return 0;
}