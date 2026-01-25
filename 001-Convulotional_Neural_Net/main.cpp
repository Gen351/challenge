#include <iostream>
#include <ctime>
#include <fstream>   // Required for checking file existence
#include <string>    // Required for std::to_string

#include "cnn/sequential.h"
#include "data_loader.h" 

// --- CONFIGURATION ---
// Set this to the name of a model to load (e.g., "fmnist0") 
// Leave it EMPTY "" to build and train a fresh model from scratch.
const std::string loadModelName = ""; 

// The base name for your saved files. 
// The code will automatically add numbers: fmnist0.cnn, fmnist1.cnn...
const std::string saveNameBase = "fmnist"; 

// ---------------------

inline bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    // Usage: ./main <epochs> <learning_rate>
    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " <epochs> <learning_rate>" << std::endl;
        return -1;
    }

    std::srand(std::time(nullptr));
    std::cout << "--- Building CNN Model ---" << std::endl;

    Sequential model;

    // 1. Build OR Load Model
    if (!loadModelName.empty()) {
        // --- LOAD EXISTING MODEL ---
        // This assumes the file is in ./trained_cnns/ (managed by ModelIO)
        std::cout << "Loading model: " << loadModelName << "..." << std::endl;
        DataLoader::ModelIO::loadModel(model, loadModelName);
    } else {
        // --- BUILD NEW MODEL ---
        std::cout << "Initializing new architecture..." << std::endl;
        
        int filters = 8;

        // Layer 1: Input 28x28x1 -> Output 24x24x8
        model.add(new ConvLayer(filters, 1, 5)); 
        model.add(new ActivationLayer(ActivationType::ReLU));

        // Layer 2: Input 24x24x8 -> Output 12x12x8
        model.add(new PoolingLayer(PoolType::MAX, 2));

        // Layer 3: Input 12x12x8 -> Output 10x10x4 
        model.add(new ConvLayer(filters / 2, filters, 3)); 
        model.add(new ActivationLayer(ActivationType::ReLU));

        // Layer 4: Input 10x10x4 -> Output 5x5x4
        model.add(new PoolingLayer(PoolType::AVERAGE, 2));

        // Layer 5: Dense (4 filters * 5 width * 5 height = 100 inputs)
        model.add(new DenseLayer(100, 10));
        model.add(new ActivationLayer(ActivationType::SoftMax));
    }

    // 2. Load Data
    std::string trainPath = "./cnn-data/mnist/fashion-mnist_train.csv";
    std::string testPath  = "./cnn-data/mnist/fashion-mnist_test.csv";

    try {
        // Load full dataset (-1 limit)
        auto trainData = DataLoader::loadFashionMNIST(trainPath, -1); 
        auto testData  = DataLoader::loadFashionMNIST(testPath, -1);
        
        // 3. Train
        int epochs = std::stoi(argv[1]);
        float lr = std::stof(argv[2]);

        // Only train if we have epochs (allows loading just for evaluation if you pass 0 epochs)
        if (epochs > 0) {
            DataLoader::train(model, trainData, epochs, lr);
        }

        // 4. Test & Visualize
        DataLoader::evaluate(model, testData, 2);

        // 5. Smart Save with Running Number
        // We know ModelIO uses "./trained_cnns/" and appends ".cnn"
        int version = 0;
        std::string finalName;
        
        while (true) {
            // Create candidate name: fmnist0, fmnist1, ...
            std::string candidateName = saveNameBase + std::to_string(version);
            
            // Check full path existence
            std::string fullPathToCheck = "./trained_cnns/" + candidateName + ".cnn";
            
            if (!fileExists(fullPathToCheck)) {
                // Found a free slot!
                finalName = candidateName;
                break;
            }
            version++;
        }

        std::cout << "Saving training result..." << std::endl;
        DataLoader::ModelIO::saveModel(model, finalName);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}