#pragma once

#include<vector>
#include<memory> 

#include "conv_layer.h"
#include "dense_layer.h"
#include "pooling_layer.h"

class Sequential {
    std::vector<std::unique_ptr<Layer>> layers;

public:

    Tensor predict(const Tensor& input) {
        Tensor current = input;

        for(auto& layer : layers) layer->predict();

        for(auto& layer : layers) {
            current = layer->forward(current);
        }

        return current;
    }


    void train(const size_t epochs,
                const std::vector<Matrix<float>> trainData,
                const std::vector<Matrix<float>> targetValues,
                const float learningRate)
    {
        if(trainData.size() != targetValues.size()) {
            throw std::runtime_error("Train_data size != Target_values size");
        }

        for(auto& layer : layers) layer->train();

        for(size_t epoch = 0; epoch < epochs; epoch++) {
            
        }
    }

private:
    Tensor feedForward(const Tensor& input) {
        layers[0]->forward(input);

        for(auto& layer : layers) {

        }
    }


};