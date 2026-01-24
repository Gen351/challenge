#pragma once

#include "abstract_layer.h"

enum class PoolType {
    MAX,
    AVERAGE
};


class PoolingLayer : public Layer {

    PoolType poolType;
    size_t poolSize;

public:

    PoolingLayer(PoolType initPoolType=PoolType::MAX, size_t initPoolSize=2)
        : poolType(initPoolType)
        , poolSize(initPoolSize < 2 ? 2 : initPoolSize) {}

    Tensor forward(const Tensor& input) override {
        Tensor output(input.featureMaps.size());
        if(poolType == PoolType::MAX) {
            for(size_t i = 0;i < input.featureMaps.size(); i++) {
                output.featureMaps[i] = maxPool(input.featureMaps[i]);
            }
        } else if(poolType == PoolType::AVERAGE) {
            for(size_t i = 0;i < input.featureMaps.size(); i++) {
                output.featureMaps[i] = avePool(input.featureMaps[i]);
            }
        }

        return output;
    }

    Tensor backward(const Tensor& output) override {
        return output;
    }

private:

    Matrix<float> maxPool(const Matrix<float>& channel) {
        Matrix<float> output(channel.rows() / poolSize, channel.cols() / poolSize);

        for(size_t i = 0; i * poolSize + poolSize <= channel.rows(); i++) {
            for(size_t j = 0; j * poolSize + poolSize <= channel.cols(); j++) {

                float max = channel[i * poolSize][j * poolSize];

                for(size_t x = 0; x < poolSize; x++) {
                    for(size_t y = 0; y < poolSize; y++) {
                        if(max < channel[i * poolSize + x][j * poolSize + y]) {
                            max = channel[i * poolSize + x][j * poolSize + y];
                        }
                    }
                }

                output[i][j] = max;
            }
        }

        return output;
    }


    Matrix<float> avePool(const Matrix<float>& channel) {
        Matrix<float> output(channel.rows() / poolSize, channel.cols() / poolSize);

        for(size_t i = 0; i * poolSize + poolSize <= channel.rows(); i++) {
            for(size_t j = 0; j * poolSize + poolSize <= channel.cols(); j++) {

                float sum = 0.0f;

                for(size_t x = 0; x < poolSize; x++) {
                    for(size_t y = 0; y < poolSize; y++) {
                        sum += channel[i * poolSize + x][j * poolSize + y];
                    }
                }

                output[i][j] = sum / (poolSize * poolSize);
            }
        }

        return output;
    }

};