#include <iostream>
#include <ctime>
#include "cnn/sequential.h"
#include "data_loader.h" 

int main() {
    // Random seed is crucial for weight initialization
    std::srand(std::time(nullptr));

    std::cout << "--- Building CNN Model ---" << std::endl;

    // 1. Build Model
    Sequential model;
    
    // Layer 1: Conv 
    // Input: 28x28 (1 channel) -> Filters: 2 (3x3) -> Output: 26x26 (2 channels)
    model.add(new ConvLayer(2, 1, 3)); 
    model.add(new ActivationLayer(ActivationType::ReLU));
    
    // Layer 2: Max Pool
    // Input: 26x26 -> Pool: 2x2 -> Output: 13x13
    model.add(new PoolingLayer(PoolType::MAX, 2));

    // Layer 3: Dense
    // Flatten Input: 2 channels * 13 * 13 = 338 inputs
    // Output: 10 classes (Fashion MNIST categories)
    model.add(new DenseLayer(338, 10)); 
    model.add(new ActivationLayer(ActivationType::SoftMax));

    // 2. Load Data
    // Ensure these paths are correct relative to your executable location
    std::string trainPath = "./cnn-data/mnist/fashion-mnist_train.csv";
    std::string testPath  = "./cnn-data/mnist/fashion-mnist_test.csv";

    try {
        // Load smaller subset for quick debugging (e.g., 2000 train, 100 test)
        // Set limit to -1 to load the full dataset later.
        auto trainData = DataLoader::loadFashionMNIST(trainPath, 2000); 
        auto testData  = DataLoader::loadFashionMNIST(testPath, 20);

        // 3. Train
        // Train for 5 epochs with Learning Rate 0.005
        DataLoader::train(model, trainData, 5, 0.005f);

        // 4. Test & Visualize
        // The last argument '20' means "Visualize the first 20 images"
        DataLoader::evaluate(model, testData, 20);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}