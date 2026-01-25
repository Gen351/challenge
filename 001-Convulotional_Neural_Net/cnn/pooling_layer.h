#pragma once

#include "abstract_layer.h"

#include<limits>

enum class PoolType {
    MAX,
    AVERAGE
};

typedef Matrix<float> (*poolFunc)(const Matrix<float>&);
typedef Matrix<float> (*backwardPoolFunc)(const Matrix<float>&, const Matrix<float>&);


class PoolingLayer : public Layer {

    size_t poolSize;
    PoolType poolType;

    Tensor lastInput;

public:

    PoolingLayer(PoolType initPoolType=PoolType::MAX, size_t initPoolSize=2)
        : poolType(initPoolType)
        , poolSize(initPoolSize < 2 ? 2 : initPoolSize) {
    }

    Tensor forward(const Tensor& input) override {
        if(training) {
            lastInput = input;
        }
        
        Tensor output;

        if (poolType == PoolType::MAX) {
            for(size_t i = 0; i < input.featureMaps.size(); i++) {
                output.addFeatureMap(maxPool(input.featureMaps[i]));
            }
        } else {
            for(size_t i = 0; i < input.featureMaps.size(); i++) {
                output.addFeatureMap(avePool(input.featureMaps[i]));
            }
        }   

        return output;
    }

    Tensor backward(const Tensor& gradientOutput) override {
        Tensor gradientInput;

        if (poolType == PoolType::MAX) {
            for(size_t i = 0; i < gradientOutput.featureMaps.size(); i++) {
                gradientInput.addFeatureMap(
                    backwardMaxPool(lastInput.featureMaps[i], gradientOutput.featureMaps[i])
                );
            }
        } else {
            for(size_t i = 0; i < gradientOutput.featureMaps.size(); i++) {
                gradientInput.addFeatureMap(
                    backwardAvePool(lastInput.featureMaps[i], gradientOutput.featureMaps[i])
                );
            }
        }
        
        return gradientInput;
    }

    void update(float learningRate) override {
        // No weights to update in a pooling layer!
    }

    // by Gemini 3.0 Pro
    std::string getType() const override { return "POOL"; }

    void save(std::ofstream& file) const override {
        // Save Config: Type (as int) and Size
        file << (int)poolType << " " << poolSize << "\n";
    }

    void load(std::ifstream& file) override {
        // Pooling has no trained weights to load.
        // The configuration was handled by the Factory to create this object.
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


    Matrix<float> backwardMaxPool(const Matrix<float>& inputMap, const Matrix<float>& gradientMap) {
        Matrix<float> result(inputMap.rows(), inputMap.cols(), 0.0f);

        for(size_t i = 0; i < gradientMap.rows(); i++) {
            for(size_t j = 0; j < gradientMap.cols(); j++) {
                
                float maxVal = std::numeric_limits<float>::lowest();
                size_t maxRow = i * poolSize;
                size_t maxCol = j * poolSize;

                // We must find the exact index of the max value again
                for(size_t x = 0; x < poolSize; x++) {
                    for(size_t y = 0; y < poolSize; y++) {
                        size_t currentRow = i * poolSize + x;
                        size_t currentCol = j * poolSize + y;
                        
                        if(currentRow < inputMap.rows() && currentCol < inputMap.cols()) {
                            if(inputMap[currentRow][currentCol] > maxVal) {
                                maxVal = inputMap[currentRow][currentCol];
                                maxRow = currentRow;
                                maxCol = currentCol;
                            }
                        }
                    }
                }
                result[maxRow][maxCol] += gradientMap[i][j];
            }
        }
        return result;
    }

    Matrix<float> backwardAvePool(const Matrix<float>& inputMap, const Matrix<float>& gradientMap) {
        Matrix<float> result(inputMap.rows(), inputMap.cols(), 0.0f);
        float distributedError = 1.0f / (poolSize * poolSize);

        for(size_t i = 0; i < gradientMap.rows(); i++) {
            for(size_t j = 0; j < gradientMap.cols(); j++) {
                
                float grad = gradientMap[i][j] * distributedError;

                for(size_t x = 0; x < poolSize; x++) {
                    for(size_t y = 0; y < poolSize; y++) {
                         size_t r = i * poolSize + x;
                         size_t c = j * poolSize + y;
                         if(r < result.rows() && c < result.cols()) {
                             result[r][c] = grad;
                         }
                    }
                }
            }
        }
        return result;
    }


};