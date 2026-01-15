#include<iostream>

#include<vector>

#include "net.h"
#include "./dataset/network_loader.h"

#include<random>
#include<time.h>

#include <cmath>

// const std::string& MNIST_TRAINED_NEURAL_NETWORK = "./dataset/FASHION_MNIST/NNs/fashion_mnist0.nn";
// const std::string& MNIST_TEST_DATA_PATH = "./dataset/FASHION_MNIST/fashion-mnist_test.csv";
const std::string& MNIST_TRAINED_NEURAL_NETWORK = "./dataset/MNIST/NNs/mnist3.nn";
const std::string& MNIST_TEST_DATA_PATH = "./dataset/MNIST/mnist_test.csv";

void setDatas(const std::vector<std::string> dataset, 
                std::vector<std::vector<float>>& testingData,
                std::vector<std::vector<float>>& targetValues)
{
    const int PIXEL_MAX_VAL = 255;
    // MNIST pixl count (28 x 28)
    const int firstLayerInputSize = 784;

    for (int i = 1; i < dataset.size(); i++) {
        std::vector<float> target(10, 0.0f);
        
        std::istringstream lineStream(dataset[i]);
        std::string token;
        if (!std::getline(lineStream, token, ',')) {
            throw std::runtime_error("Empty line or missing label at line " + std::to_string(i));
        }
        int label = std::stoi(token);
        if (label >= 10) {
            throw std::runtime_error("Invalid label >= 10 at line " + std::to_string(i));
        }
        target[label] = 1.0f;
        targetValues.emplace_back(target);
        
        // Read the rest of the tokens as input data
        std::vector<float> data;
        while (std::getline(lineStream, token, ',')) {
            float inputData = std::stof(token) / PIXEL_MAX_VAL;
            data.push_back(inputData);
        }
        
        if (data.size() != firstLayerInputSize) {
            throw std::runtime_error(
                "testingData[" + std::to_string(i) + "].size() does not match required size "
                + std::to_string(firstLayerInputSize) + ": " + std::to_string(data.size())
            );
        }
        testingData.emplace_back(data);
        
        if(i % 10000 == 0) {
            std::cout << i << '\n';
        }
    }
}

void printDigit(const std::vector<float>& pixels) {
    std::cout << "\n--- IMAGE PREVIEW ---\n";
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Get pixel value (0.0 to 1.0)
            float val = pixels[y * 28 + x];
            
            // Convert to 0-255 integer
            int intensity = static_cast<int>(val * 255.0f);
            
            // ANSI Escape Code for Background Color: \x1b[48;2;R;G;Bm
            // We use the same value for R, G, and B to get grayscale.
            // We print two spaces "  " because terminal characters are usually tall rectangles.
            std::cout << "\x1b[48;2;" << intensity << ";" << intensity << ";" << intensity << "m  ";
        }
        // Reset color at end of row
        std::cout << "\x1b[0m\n"; 
    }
    std::cout << "\x1b[0m---------------------\n";
}


int main(int argc, char* argv[]) {
    srand(time(0));

    Net nn = NetworkLoader::load(MNIST_TRAINED_NEURAL_NETWORK);


    std::vector<std::string> dataset = DatasetLoader::load(MNIST_TEST_DATA_PATH);    
    std::vector<std::vector<float>> testingData;
    std::vector<std::vector<float>> targetValues;
    setDatas(dataset, testingData, targetValues);
    
    // int correctCounter = 0;
    // for(int i = 0; i < testingData.size(); i++) {
    //     const auto prediction = nn.predict(testingData[i]);

    //     int predictHigh = 0;
    //     int targetHigh = 0;
    //     for(int j = 1; j < 10; j++) {
    //         if(prediction[j] > prediction[predictHigh]) predictHigh = j;
    //         if(targetValues[i][j] > targetValues[i][targetHigh]) targetHigh = j;
    //     }

    //     correctCounter += (targetHigh == predictHigh);
    // }

    // float accuracy = (float)correctCounter / testingData.size();
    // printf("Accuracy: %.8f", accuracy * 100);

    int randomIndex = rand() % testingData.size() - 1;

    printDigit(testingData[randomIndex]);
    const auto pred = nn.predict(testingData[randomIndex]);
    int max = 0;
    for(int i = 1; i < pred.size(); i++) {
        if(pred[i] > pred[max]) max = i;
    }
    std::cout << "Prediction: " << max << '\n';

    return argc;
}