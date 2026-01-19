#pragma once

#include<stdexcept>
#include<vector>
    
template<typename T>
struct Matrix {
    size_t row;
    size_t col;

    std::vector<T> data;

    Matrix() 
        : row(0), col(0), data() {}

    Matrix(size_t r, size_t c, T init = T{}) 
        : row(r), col(c), data(r * c, init) {}

    
    T* operator[](size_t r) {
        if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
        return &data[r * col];
    }
    const T* operator[](size_t r) const {
        if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
        return &data[r * col];
    }

    T& operator()(size_t r, size_t c) {
        if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
        return data[r * col + c];
    }
    const T& operator()(size_t r, size_t c) const {
        if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
        return data[r * col + c];
    }

    size_t rows() const { return row; }
    size_t cols() const { return col; }
};