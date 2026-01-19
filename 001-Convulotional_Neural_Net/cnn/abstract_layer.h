#pragma once

#include<math.h>

#include "tensor.h"

class Layer {

public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& input) = 0;


    float sigmoid(float x) {
        return (1.0f / (1.0f + std::exp(-x)));
    }

    float sigmoidDerivative(float val) {
        return (val * (1.0f - val));
    }

    float ReLU(float val) {
        return val * (val > 0);
    }
};