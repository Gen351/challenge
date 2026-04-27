#include <iostream>
#include <ctime>
#include <fstream>   // Required for checking file existence
#include <string>    // Required for std::to_string

#include "cnn/sequential.h"
#include "data_loader.h" 

// --- CONFIGURATION ---
// Set this to the name of a model to load (e.g., "fmnist0") 
// Leave it EMPTY "" to build and train a fresh model from scratch.
const std::string loadModelName = "Smarter_Training_CAPCAPDADA0"; 

// The base name for your saved files. 
// The code will automatically add numbers: fmnist0.cnn, fmnist1.cnn...
const std::string saveNameBase = "Smarter_Training_CAPCAPDADA";

// ---------------------

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
        
        int filters = 16;
        int currentSize = 28;

        // Layer 1: 28x28x1 -> 24x24x32
        model.add(new ConvLayer(filters, 1, 5)); 
        model.add(new ActivationLayer(ActivationType::ReLU));
        currentSize = 24; 

        // Layer 2: 24x24x32 -> 12x12x32
        model.add(new PoolingLayer(PoolType::MAX, 2));
        currentSize = 12;

        // Layer 3: 12x12x32 -> 10x10x64
        int prevFilters = filters;
        filters = 32; 
        model.add(new ConvLayer(filters, prevFilters, 3)); 
        model.add(new ActivationLayer(ActivationType::ReLU));
        currentSize = 10; 

        // Layer 4: 10x10x64 -> 5x5x64
        model.add(new PoolingLayer(PoolType::AVERAGE, 2));
        currentSize = 5;

        // --- NEW HIDDEN DENSE SECTION ---
        // Layer 5: Flatten (800) -> Hidden (128)
        int flattenedSize = currentSize * currentSize * filters; // 1600
        int hiddenNeurons = 128; 
        model.add(new DenseLayer(flattenedSize, hiddenNeurons));
        model.add(new ActivationLayer(ActivationType::ReLU)); 

        // Layer 6: Hidden (128) -> Output (10)
        model.add(new DenseLayer(hiddenNeurons, 10));
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
            DataLoader::train(model, trainData, epochs, lr, true);
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
                // Found a free slot
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