#pragma once

#include "abstract_layer.h"
#include "./conv_layer/filter.h"

#include "tensor.h"

class ConvLayer : public Layer {

    std::vector<Filter> filters;
    std::vector<float> biases;

public:
    ConvLayer() = default;
    ConvLayer(size_t filterCount, size_t channelCount=1, size_t kernelDimensions=3)
        : filters(filterCount, Filter(channelCount, kernelDimensions))
        , biases(filterCount, 0) {}

    Tensor forward(const Tensor& input) override {
        if (input.featureMaps.empty()) return Tensor();
        if (input.featureMaps.size() != filters[0].channels.size()) {
            throw std::runtime_error("Input channels do not match filter depth!");
        }

        Tensor output;

        for(size_t f = 0; f < filters.size(); f++) {
            Matrix<float> total = convolve(input.featureMaps[0], filters[f].channels[0]);

            for(size_t c = 1; c < input.featureMaps.size(); c++) {
                Matrix<float> channelResult = convolve(input.featureMaps[c], filters[f].channels[c]);
                total = MatrixOp::add(total, channelResult);
            }

            // Adding the bias
            float bias = biases[f];
            for(size_t i = 0; i < total.data.size(); i++) {
                total.data[i] = ReLU(total.data[i] + bias);
            }

            output.addFeatureMap(total);
        }

        return output;
    }


    Matrix<float> convolve(const Matrix<float>& featureMap, const Kernel& kernel) {
        if(featureMap.rows() < kernel.data.rows() || featureMap.cols() < kernel.data.cols()) {
            throw std::runtime_error("Covolve: featureMap should be: featureMap >= (3 x 3)");
        }

        Matrix<float> convolved((featureMap.rows() - kernel.data.rows()) + 1, (featureMap.cols() - kernel.data.cols()) + 1);

        for(size_t i = 0; i < convolved.rows(); i++) {
            for(size_t j = 0; j < convolved.cols(); j++) {
                // create patch
                float patchSum = 0;
                for(size_t x = 0; x < kernel.data.rows(); x++) {
                    for(size_t y = 0; y < kernel.data.cols(); y++) {
                        patchSum += featureMap[i+x][j+y] * kernel.data[x][y];
                    }
                }

                convolved[i][j] = patchSum;
            }
        }

        return convolved;
    }



    /// @brief Converts a Matrix into a "flat" Vector or a 1-D array
    /// @brief ... using std::vector's assignment operator. 
    /// @param A - Matrix (n x m)
    /// @return V - Vector (n * m)
    Vector<float> flatten(const Matrix<float>& A) {
        Vector<float> V;
        V.data = A.data;
        return V;
    }

};