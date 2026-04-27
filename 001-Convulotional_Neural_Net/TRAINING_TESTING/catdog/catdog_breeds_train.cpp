#include <iostream>
#include <ctime>
#include <fstream>
#include <string>

// 1. Include your Core Headers
#include "../../cnn/sequential.h"
#include "../../cnn/tensor.h"
#include "../../cnn/matrix_op/matrix.hpp"

// 2. Include Layer Implementations
#include "../../cnn/conv_layer.h"
#include "../../cnn/pooling_layer.h"
#include "../../cnn/dense_layer.h"
#include "../../cnn/activation_layer.h"

// 3. Include Loaders
#include "../../image_loader.h" // For loading images
#include "../../data_loader.h"  // <--- ADDED THIS (Contains ModelIO for saving/loading)

// --- CONFIGURATION ---
const std::string csvPath = "../../cnn-data/catdog/breeds_shuffled_train.csv"; // Make sure path matches your folder structure
const std::string saveNameBase = "cat_dog_breedsRGB";

// Set this to "" to train from scratch, or "modelName0" to resume
const std::string loadModelName = "";

// Image Settings
const int IMG_SIZE = 100;
const int CHANNELS = 3; 
const int CLASSES = 37;
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
        data = ImageLoader::loadFromCSV(csvPath, IMG_SIZE, IMG_SIZE, CHANNELS, CLASSES, -1);
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
            DataLoader::ModelIO::loadModel(model, loadModelName, "../../trained_cnns/");
        } catch(const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "--- Initializing RGB Architecture ---" << std::endl;

        // ============================================================
        // [ARCHITECTURE DESIGN AREA]
        // Input: 100 x 100 x 3 (RGB)
        // ============================================================

        // --- Layer 1 ---
        // 100x100 -> Conv(3x3) -> 98x98 -> Pool(2x2) -> 49x49
        model.add(new ConvLayer(32, CHANNELS, 3));
        model.add(new ActivationLayer(ActivationType::LeakyReLU));
        model.add(new PoolingLayer(PoolType::MAX, 2));

        // --- Layer 2 ---
        // 49x49 -> Conv(3x3) -> 47x47 -> Pool(2x2) -> 23x23
        model.add(new ConvLayer(64, 32, 3));
        model.add(new ActivationLayer(ActivationType::LeakyReLU));
        model.add(new PoolingLayer(PoolType::MAX, 2));

        // ... Layer 3 ...
        model.add(new ConvLayer(128, 64, 3));
        model.add(new ActivationLayer(ActivationType::LeakyReLU));
        model.add(new PoolingLayer(PoolType::MAX, 2));
        
        // ==========================================
        // DYNAMIC FLATTEN CALCULATION
        // ==========================================
        // 1. Create a "Dummy" Input matching your image dimensions
        Tensor dummyInput;
        for (int c = 0; c < CHANNELS; c++) {
            dummyInput.addFeatureMap(Matrix<float>(IMG_SIZE, IMG_SIZE));
        }

        // 2. Pass it through the layers added so far
        Tensor dummyOutput = dummyInput;
        for (auto& layer : model.getLayers()) { // (Assuming 'layers' is public in Sequential)
            dummyOutput = layer->forward(dummyOutput);
        }

        // 3. Calculate the total size of the resulting 3D Tensor
        int outChannels = dummyOutput.featureMaps.size();
        int outRows = dummyOutput.featureMaps[0].rows();
        int outCols = dummyOutput.featureMaps[0].cols();
        int flattenSize = outChannels * outRows * outCols;

        std::cout << "--- Auto-calculated Flatten Size: " << flattenSize << " ---" << std::endl;

        // ==========================================

        // --- Flattening ---
        // Now use the dynamic variable instead of a hardcoded number!
        model.add(new DenseLayer(flattenSize, 640)); 
        model.add(new ActivationLayer(ActivationType::LeakyReLU));

        // Output Layer
        model.add(new DenseLayer(640, CLASSES));
        model.add(new ActivationLayer(ActivationType::SoftMax));
        // ============================================================
    }

    // 3. Train
    int epochs = std::stoi(argv[1]);
    float lr = std::stof(argv[2]);

    // Using your 'trainSmart' from Sequential class
    model.trainSmartCrossEntropy(epochs, data.inputs, data.targets, lr, false, 30);

    // 4. Smart Save
    // This loop finds the next available filename (e.g., FlowerAmpota_RGB0, RGB1, RGB2...)
    int version = 0;
    std::string finalName;
    while (true) {
        std::string candidateName = "catdog/" + saveNameBase + std::to_string(version);
        // Check if file exists in the folder defined in ModelIO::DIR
        if (!fileExists("../../trained_cnns/" + candidateName + ".cnn")) {
            finalName = candidateName;
            break;
        }
        version++;
    }

    std::cout << "Saving model to: " << finalName << std::endl;
    
    // FIXED: Using ModelIO from data_loader.h
    DataLoader::ModelIO::saveModel(model, finalName, "../../trained_cnns/"); 

    return 0;
}