#pragma once

#include<math.h>

#include<fstream>

#include "tensor.h"

enum class LayerType {
    CONV,
    POOL,
    ACT,
    DENSE,
    FLATTEN,
    UNKNOWN
};

class Layer {

protected:

    bool training = false;
    
public:

    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& input) = 0;
    virtual void update(float learningRate) = 0;

    virtual void save(std::ofstream& file) const = 0;
    virtual void load(std::ifstream& file) = 0;
    virtual std::string getType() const = 0;

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