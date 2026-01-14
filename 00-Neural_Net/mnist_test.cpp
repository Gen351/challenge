#include<iostream>

#include<vector>

#include "net.h"
#include "./dataset/network_loader.h"

const std::string& MNIST_TRAINED_NEURAL_NETWORK = "./dataset/MNIST/NNs/mnist3.nn";
const std::string& MNIST_TEST_DATA_PATH = "./dataset/MNIST/mnist_train.csv";

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

int main(int argc, char* argv[]) {
    
    Net nn = NetworkLoader::load(MNIST_TRAINED_NEURAL_NETWORK);


    std::vector<std::string> dataset = DatasetLoader::load(MNIST_TEST_DATA_PATH);    
    std::vector<std::vector<float>> testingData;
    std::vector<std::vector<float>> targetValues;
    setDatas(dataset, testingData, targetValues);
    
    int correctCounter = 0;
    for(int i = 0; i < testingData.size(); i++) {
        const auto prediction = nn.predict(testingData[i]);

        int predictHigh = 0;
        int targetHigh = 0;
        for(int j = 1; j < 10; j++) {
            if(prediction[j] > prediction[predictHigh]) predictHigh = j;
            if(targetValues[i][j] > targetValues[i][targetHigh]) targetHigh = j;
        }

        correctCounter += (targetHigh == predictHigh);
    }

    float accuracy = (float)correctCounter / testingData.size();
    printf("Accuracy: %.8f", accuracy * 100);

    return argc;
}