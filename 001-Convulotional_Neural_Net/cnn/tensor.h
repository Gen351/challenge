#pragma once

#include "./matrix_op/matrix_op.hpp"

struct Tensor {

    std::vector<Matrix<float>> featureMaps;

public:

    Tensor() = default;
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor(size_t featureMapCount)
        : featureMaps(featureMapCount)
    {}
    Tensor(const Tensor& other) = default;
    Tensor(std::vector<Matrix<float>> initFeatureMaps)
        : featureMaps(initFeatureMaps)
    {}

    void addFeatureMap(const Matrix<float>& featureMap) {
        featureMaps.push_back(featureMap);
    }

};