#include<iostream>
#include<fstream>

#include<vector>

#include "net.h"
#include "./dataset/network_formatter.h"
#include "./dataset/network_saver.h"
#include "./dataset/dataset_loader.h"

const std::string& MNIST_TRAIN_DATA_PATH = "./dataset/MNIST/mnist_train.csv";
const std::string& MNIST_NETWORK_SAVE_PATH = "./dataset/MNIST/NNs/mnist";

int main(int argc, char* argv[]) {
    
    const int PIXEL_MAX_VAL = 255;
    // MNIST pixl count (28 x 28)
    const int firstLayerInputSize = 784;
    
    /* ============================== */
    /* C O N T R O L S      S T A R T */
    /* ============================== */
    const int epochs = 5;
    const float learningRate = 0.03f;
    std::vector<int> layerSizes = {firstLayerInputSize, 64, 10};
    const bool SAVE = true;
    /* ============================== */
    /* C O N T R O L S          E N D */
    /* ============================== */
    

    Net nn(layerSizes);



    // nn.DEBUG();
    // to show the current path
    // std::cout << argv[0] << '\n';
    
    std::vector<std::string> dataset = DatasetLoader::load(MNIST_TRAIN_DATA_PATH);
    
    std::vector<std::vector<float>> trainingData;
    std::vector<std::vector<float>> targetValues;
    
    // ignore dataset[0] <-- (label)
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
                "trainingData[" + std::to_string(i) + "].size() does not match required size "
                + std::to_string(firstLayerInputSize) + ": " + std::to_string(data.size())
            );
        }
        trainingData.emplace_back(data);
        
        if(i % 10000 == 0) {
            std::cout << i << ',';
        }
    }
    
    
    /* ============================== */
    /* T R A I N            S T A R T */
    /* ============================== */
    std::cout << "\n\nTraining\n";
    nn.train(epochs, trainingData, targetValues, learningRate, "print please");
    std::cout << "Training Done\n";
    /* ============================== */
    /* T R A I N                E N D */
    /* ============================== */
    


    /* ============================== */
    /* F O R M A T T I N G  S T A R T */
    /* ============================== */
    std::cout << "NeuralNet -> vector<string>\n";
    std::vector<std::string> formattedNetwork = NetworkFormatter::formatNetwork(nn);
    std::cout << "NeuralNet -> vector<string> Done\n";
    /* ============================== */
    /* F O R M A T T I N G      E N D */
    /* ============================== */
    

    /* ============================== */
    /* S A V E              S T A R T */
    /* ============================== */
    std::cout << "saving (NeuralNet)vector<string>\n";
    NetworkSaver::saveNetwork(formattedNetwork, MNIST_NETWORK_SAVE_PATH);
    std::cout << "saving (NeuralNet)vector<string> Done\n";
    /* ============================== */
    /* S A V E                  E N D */
    /* ============================== */
    

    const int TEST = 123;
    const auto nntest = nn.predict(trainingData[TEST]);

    for(int i = 0; i < 10; i++) {
        printf("%.10f\t\t%1f\n", nntest[i], targetValues[TEST][i]);
    }

    return argc;
}