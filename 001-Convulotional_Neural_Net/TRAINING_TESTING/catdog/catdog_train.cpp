#include <iostream>
#include <ctime>
#include <fstream>
#include <string>

// 1. Include your Core Headers
#include "cnn/sequential.h"
#include "cnn/tensor.h"
#include "cnn/matrix_op/matrix.hpp"

// 2. Include Layer Implementations
#include "cnn/conv_layer.h"
#include "cnn/pooling_layer.h"
#include "cnn/dense_layer.h"
#include "cnn/activation_layer.h"

// 3. Include Loaders
#include "image_loader.h" // For loading images
#include "data_loader.h"  // <--- ADDED THIS (Contains ModelIO for saving/loading)

// --- CONFIGURATION ---
const std::string csvPath = "cnn-data/catdog/CatDog.csv"; // Make sure path matches your folder structure
const std::string saveNameBase = "cat_dogRGB";

// Set this to "" to train from scratch, or "modelName0" to resume
const std::string loadModelName = "";

// Image Settings
const int IMG_SIZE = 64;
const int CHANNELS = 3; 
const int CLASSES = 2;
// ---------------------

// Simple helper to check file existence before overwriting
inline bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " <epochs> <learning_rate>" << std::endl;
        return -1;
    }

    std::srand(std::time(nullptr));

    // 1. Load Data
    std::cout << "--- Loading [ " << csvPath << " ] Dataset ---" << std::endl;
    ImageLoader::Dataset data;
    try {
        data = ImageLoader::loadFromCSV(csvPath, IMG_SIZE, IMG_SIZE, CHANNELS, CLASSES);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error Loading Data: " << e.what() << std::endl;
        return 1;
    }

    // 2. Build Model
    Sequential model;

    if (!loadModelName.empty()) {
        std::cout << "Loading model: " << loadModelName << "..." << std::endl;
        try {
            // FIXED: Using ModelIO from data_loader.h
            DataLoader::ModelIO::loadModel(model, loadModelName);
        } catch(const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "--- Initializing RGB Architecture ---" << std::endl;

        // ============================================================
        // [ARCHITECTURE DESIGN AREA]
        // Input: 64 x 64 x 3 (RGB)
        // ============================================================

        // --- Layer 1 ---
        // 64x64 -> 62x62 -> Pool to 31x31
        model.add(new ConvLayer(20, CHANNELS, 3)); 
        model.add(new ActivationLayer(ActivationType::LeakyReLU));
        model.add(new PoolingLayer(PoolType::MAX, 2));

        // --- Layer 2 ---
        // 31x31 -> 29x29 -> Pool to 14x14
        model.add(new ConvLayer(40, 20, 3));
        model.add(new ActivationLayer(ActivationType::LeakyReLU));
        model.add(new PoolingLayer(PoolType::MAX, 2));

        // --- Layer 3 ---
        // 14x14 -> 12x12 -> Pool to 6x6
        model.add(new ConvLayer(80, 40, 3));
        model.add(new ActivationLayer(ActivationType::ReLU));
        model.add(new PoolingLayer(PoolType::AVERAGE, 2));

        // --- Flattening ---
        // 6 * 6 * 80 = 2880 inputs
        model.add(new DenseLayer(2880, 320)); 
        model.add(new ActivationLayer(ActivationType::ReLU));

        // Output Layer
        model.add(new DenseLayer(320, CLASSES));
        model.add(new ActivationLayer(ActivationType::SoftMax));
        // ============================================================
    }

    // 3. Train
    int epochs = std::stoi(argv[1]);
    float lr = std::stof(argv[2]);

    // Using your 'trainSmart' from Sequential class
    model.trainSmart(epochs, data.inputs, data.targets, lr, true);

    // 4. Smart Save
    // This loop finds the next available filename (e.g., FlowerAmpota_RGB0, RGB1, RGB2...)
    int version = 0;
    std::string finalName;
    while (true) {
        std::string candidateName = "catdog/" + saveNameBase + std::to_string(version);
        // Check if file exists in the folder defined in ModelIO::DIR
        if (!fileExists("./trained_cnns/" + candidateName + ".cnn")) {
            finalName = candidateName;
            break;
        }
        version++;
    }

    std::cout << "Saving model to: " << finalName << std::endl;
    
    // FIXED: Using ModelIO from data_loader.h
    DataLoader::ModelIO::saveModel(model, finalName); 

    return 0;
}