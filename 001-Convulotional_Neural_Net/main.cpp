#include <iostream>
#include <vector>
#include <ctime>

// Include your headers
#include "./cnn/sequential.h"

// Helper to generate a dummy 5x5 image with a vertical line
Matrix<float> createVerticalLineImage() {
    Matrix<float> m(5, 5, 0.0f);
    for(int i=0; i<5; i++) m(i, 2) = 1.0f; // Line down the middle
    return m;
}

// Helper to generate a dummy 5x5 image with a horizontal line
Matrix<float> createHorizontalLineImage() {
    Matrix<float> m(5, 5, 0.0f);
    for(int j=0; j<5; j++) m(2, j) = 1.0f; // Line across the middle
    return m;
}

int main() {
    // 1. Initialize Random Seed (Crucial for weights!)
    std::srand(std::time(nullptr));

    std::cout << "--- Building CNN ---" << std::endl;

    Sequential model;

    // --- ARCHITECTURE ---
    // Input Image: 5x5
    
    // Layer 1: Conv
    // Input: 1 channel (grayscale)
    // Filters: 2 filters of size 3x3
    // Output calculation: (5 - 3) + 1 = 3x3 Output Map
    model.add(new ConvLayer(2, 1, 3)); 
    
    // Layer 2: ReLU
    model.add(new ActivationLayer(ActivationType::ReLU));

    // Layer 3: Max Pooling
    // Input: 3x3
    // Pool Size: 2x2
    // Output calculation: 3 / 2 = 1 (Integer division) -> 1x1 Output
    // Note: We have 2 filters, so we have two 1x1 maps.
    model.add(new PoolingLayer(PoolType::MAX, 2));

    // Layer 4: Dense (Fully Connected)
    // Input Size calculation: (Feature Maps) * (Rows) * (Cols) 
    //                     = 2 * 1 * 1 = 2 inputs.
    // Output Size: 2 classes (Vertical vs Horizontal)
    model.add(new DenseLayer(2, 2));

    // Layer 5: Softmax (Probability distribution)
    model.add(new ActivationLayer(ActivationType::SoftMax));


    // --- DATA PREPARATION ---
    std::cout << "--- Generating Data ---" << std::endl;
    
    std::vector<Matrix<float>> trainData;
    std::vector<Matrix<float>> targets;

    // Create 100 samples
    for(int i = 0; i < 50; i++) {
        // Sample 1: Vertical Line -> Target [1, 0]
        trainData.push_back(createVerticalLineImage());
        Matrix<float> t1(1, 2); t1(0,0) = 1; t1(0,1) = 0;
        targets.push_back(t1);

        // Sample 2: Horizontal Line -> Target [0, 1]
        trainData.push_back(createHorizontalLineImage());
        Matrix<float> t2(1, 2); t2(0,0) = 0; t2(0,1) = 1;
        targets.push_back(t2);
    }


    // --- TRAINING ---
    std::cout << "--- Starting Training ---" << std::endl;
    // Train for 50 epochs with learning rate 0.01
    model.train(50, trainData, targets, 0.01f);


    // --- PREDICTION / TESTING ---
    std::cout << "--- Testing ---" << std::endl;

    // Test Case 1: Vertical Line
    Tensor inputV; 
    inputV.addFeatureMap(createVerticalLineImage());
    Tensor outV = model.predict(inputV);
    
    std::cout << "Input: Vertical Line | Prediction: [ " 
              << outV.featureMaps[0].data[0] << " (Vert), " 
              << outV.featureMaps[0].data[1] << " (Horz) ]" << std::endl;

    // Test Case 2: Horizontal Line
    Tensor inputH; 
    inputH.addFeatureMap(createHorizontalLineImage());
    Tensor outH = model.predict(inputH);

    std::cout << "Input: Horiz Line    | Prediction: [ " 
              << outH.featureMaps[0].data[0] << " (Vert), " 
              << outH.featureMaps[0].data[1] << " (Horz) ]" << std::endl;

    return 0;
}