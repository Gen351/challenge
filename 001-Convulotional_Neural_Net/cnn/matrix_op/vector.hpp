#pragma once

#include<stdexcept>
#include<vector>

template<typename T>
struct Vector {
    std::vector<T> data;

    Vector() 
        : data(0) {}

    Vector(size_t size) 
        : data(size) {}
    

    T& operator[](size_t index) {
        return data[index];
    }
    const T& operator[](size_t index) const {
        return data[index];
    }

    size_t size() const { return data.size(); }
};