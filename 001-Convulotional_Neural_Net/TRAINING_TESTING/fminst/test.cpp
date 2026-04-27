#include <iostream>
#include <ctime>
#include <fstream>   // Required for checking file existence
#include <string>    // Required for std::to_string

#include "cnn/sequential.h"
#include "data_loader.h" 

// --- CONFIGURATION ---
// Set this to the name of a model to load (e.g., "fmnist0") 
const std::string loadModelName = "Smarter_Training_CAPCAPDADA0";

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " <visualize_count> <only see incorrect(0,1)>" << std::endl;
        return -1;
    }
    std::string argFlag = argv[2];
    bool showIncorrect = (argFlag == "1" || argFlag == "true");


    Sequential model;

    // 1. Build OR Load Model
    if (!loadModelName.empty()) {
        // --- LOAD EXISTING MODEL ---
        // This assumes the file is in ./trained_cnns/ (managed by ModelIO)
        std::cout << "Loading model: " << loadModelName << "..." << std::endl;
        DataLoader::ModelIO::loadModel(model, loadModelName);
    } else {
        std::cout << "Empty Model!\n";
    }

    std::string testPath  = "./cnn-data/mnist/fashion-mnist_test.csv";

    try {
        // Load full dataset (-1 limit)
        auto testData  = DataLoader::loadFashionMNIST(testPath, -1);

        // Test & Visualize
        int visCount = std::stoi(argv[1]);
        DataLoader::evaluate(model, testData, visCount, showIncorrect);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}