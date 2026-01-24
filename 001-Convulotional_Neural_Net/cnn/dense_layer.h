#pragma once

#include "abstract_layer.h"

class DenseLayer : public Layer {

    Matrix<float> weights;
    std::vector<float> bias;
    Tensor lastInput;
    Tensor lastOutput;

public:

    DenseLayer(size_t inputSize=1, size_t outputSize=1)
        : weights(inputSize, outputSize)
        , bias(outputSize)
    {
        const float scale = std::sqrt(1.0f / inputSize);
        for(float& weight : weights) weight = MatrixOp::initRandFloat() * scale;
    }

    Tensor forward(const Tensor& input) {
        Matrix<float> flatMatrix(1, bias.size());

        for(size_t i = 0; i < bias.size(); i++) {
            float sum = 0;

            size_t weightRowIndex = 0;
            for(size_t j = 0; j < input.featureMaps.size(); j++) {
                for(size_t k = 0; k < input.featureMaps[j].data.size(); k++) {
                    sum += input.featureMaps[j].data[k] * weights(weightRowIndex, i);
                    weightRowIndex++;
                }
            }
            sum += bias[i];
            flatMatrix(0, i) = sum;
        }

        Tensor output;
        output.addFeatureMap(flatMatrix);
        
        return output;
    }

    Tensor backward(const Tensor& output) {





        return output;
    }

};