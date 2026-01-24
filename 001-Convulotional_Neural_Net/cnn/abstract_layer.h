#pragma once

#include<math.h>

#include "tensor.h"

class Layer {

protected:

    bool training = false;
    
public:

    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& input) = 0;

    void train() {
        training = true;
    }
    void predict() {
        training = false;
    }

    size_t caclulateInputDimension(const Tensor& input) {
        if(input.featureMaps.empty()) {
            return 0;
        }

        return input.featureMaps.size() * input.featureMaps[0].data.size();
    }
};