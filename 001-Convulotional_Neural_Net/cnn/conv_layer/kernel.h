#pragma once

#include "../matrix_op/matrix_op.hpp"

struct Kernel {
    Matrix<float> data;

    Kernel(size_t dimension=3) : data(dimension, dimension) {
        for(size_t i = 0; i < data.rows(); i++) {
            for(size_t j = 0; j < data.cols(); j++) {
                data[i][j] = MatrixOp::initRandFloat();
            }
        }
    }

    const auto begin() const {
        return data.begin();
    }
    const auto end() const {
        return data.end();
    }

    auto begin() {
        return data.begin();
    }
    auto end() {
        return data.end();
    }
};