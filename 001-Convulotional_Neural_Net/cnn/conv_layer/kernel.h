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
};