#pragma once

#include "abstract_layer.h"
#include "./conv_layer/filter.h"

class ConvLayer : public Layer {

    std::vector<Filter> filters;
    std::vector<float> biases;
    
    Tensor lastInput;
    Tensor lastOutput;

    std::vector<Filter> gradientFilters;
    std::vector<float> gradientBias;

public:

    ConvLayer() = default;
    ConvLayer(size_t filterCount, size_t channelCount=1, size_t kernelDimensions=3)
        : filters(filterCount, Filter(channelCount, kernelDimensions))
        , biases(filterCount, 0)
        , gradientFilters(filterCount, Filter(channelCount, kernelDimensions))
        , gradientBias(filterCount, 0)
    {
        const int inputs = channelCount * kernelDimensions * kernelDimensions;
        
        for(auto& filter : filters) {
            for(auto& channel : filter.channels) {
                for(float& data : channel) {
                    data = MatrixOp::XavierGlorotRandom(inputs);
                    // data = MatrixOp::initRandFloat();
                }
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.featureMaps.empty()) return Tensor();
        if (input.featureMaps.size() != filters[0].channels.size()) {
            throw std::runtime_error("Input channels do not match filter depth!");
        }
        if(training) {
            lastInput = input;
        }

        Tensor output;

        for(size_t f = 0; f < filters.size(); f++) {
            Matrix<float> total = convolve(input.featureMaps[0], filters[f].channels[0].data);

            for(size_t c = 1; c < input.featureMaps.size(); c++) {
                Matrix<float> channelResult = convolve(input.featureMaps[c], filters[f].channels[c].data);
                MatrixOp::addInPlace(total, channelResult);
            }

            // Adding the bias
            float bias = biases[f];
            for(size_t i = 0; i < total.data.size(); i++) {
                total.data[i] = total.data[i] + bias;
            }

            output.addFeatureMap(total);
        }

        lastOutput = output;
        return output;
    }

    Tensor backward(const Tensor& gradientOutput) override {
        for(size_t m = 0; m < gradientOutput.featureMaps.size(); m++) {
            float sum = 0.0f;
            for(size_t c = 0; c < gradientOutput.featureMaps[m].data.size(); c++) {
                sum += gradientOutput.featureMaps[m].data[c];
            }
            gradientBias[m] = sum;
        }

        // 2. KERNEL GRADIENT
        for(size_t f = 0; f < filters.size(); f++) {
            for(size_t c = 0; c < filters[f].channels.size(); c++) {
                gradientFilters[f].channels[c].data = convolve(lastInput.featureMaps[c], gradientOutput.featureMaps[f]);
            }
        }

        Tensor gradientInput;
        size_t inputChannels = lastInput.featureMaps.size();
        size_t kSize = filters[0].channels[0].data.rows();
        size_t paddingNeeded = kSize - 1;

        // Initialize empty feature maps for the input gradient (matching original input size)
        for(size_t c = 0; c < inputChannels; c++) {
            gradientInput.addFeatureMap(Matrix<float>(lastInput.featureMaps[c].rows(), lastInput.featureMaps[c].cols(), 0.0f));
        }

        // The "Full Convolution" Loop
        for(size_t f = 0; f < filters.size(); f++) {
            // Pad the error map so we can "expand" it back to input size
            Matrix<float> paddedError = MatrixOp::pad(gradientOutput.featureMaps[f], paddingNeeded);

            for(size_t c = 0; c < inputChannels; c++) {
                // Chain Rule: Gradient = PaddedError * RotatedKernel
                Matrix<float> rotatedKernel = MatrixOp::rotate180(filters[f].channels[c].data);
                Matrix<float> errorContribution = convolve(paddedError, rotatedKernel);

                // Accumulate the error for this channel from all filters
                MatrixOp::addInPlace(gradientInput.featureMaps[c], errorContribution);
            }
        }

        return gradientInput;
    }


    void update(float learningRate) override {
        // 1. Update Filters (Kernels)
        for(size_t f = 0; f < filters.size(); f++) {
            for(size_t c = 0; c < filters[f].channels.size(); c++) {
                
                // Get references to the Weight Matrix and the Gradient Matrix
                // Assuming filters[f].channels[c].data is your Matrix<float>
                std::vector<float>& weights = filters[f].channels[c].data.data;
                std::vector<float>& grads = gradientFilters[f].channels[c].data.data;

                // Subtract gradient from weight
                for(size_t i = 0; i < weights.size(); i++) {
                    weights[i] -= learningRate * grads[i];
                    
                    // IMPORTANT: Reset gradient to 0 for the next batch/image
                    grads[i] = 0.0f;
                }
            }
        }

        // 2. Update Biases
        for(size_t f = 0; f < biases.size(); f++) {
            biases[f] -= learningRate * gradientBias[f];
            
            // Reset bias gradient
            gradientBias[f] = 0.0f;
        }
    }


private:

    Matrix<float> convolve(const Matrix<float>& featureMap, const Matrix<float>& kernel) {
        if(featureMap.rows() < kernel.rows() || featureMap.cols() < kernel.cols()) {
            throw std::runtime_error("Covolve: featureMap should be: featureMap >= (3 x 3)");
        }
        Matrix<float> convolved((featureMap.rows() - kernel.rows()) + 1, (featureMap.cols() - kernel.cols()) + 1);

        for(size_t i = 0; i < convolved.rows(); i++) {
            for(size_t j = 0; j < convolved.cols(); j++) {
                // create patch
                float patchSum = 0;
                for(size_t x = 0; x < kernel.rows(); x++) {
                    for(size_t y = 0; y < kernel.cols(); y++) {
                        patchSum += featureMap(i+x, j+y) * kernel(x, y);
                    }
                }

                convolved[i][j] = patchSum;
            }
        }

        return convolved;
    }
};