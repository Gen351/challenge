#pragma once

#include "kernel.h"

struct Filter {
    std::vector<Kernel> channels;

    Filter() = default;
    Filter(size_t channelCount) : channels(channelCount) {}
    Filter(size_t channelCount, size_t kernelDimensions) 
        : channels(channelCount) {
        for(auto& kernel : channels) {
            kernel = Kernel(kernelDimensions);
        }
    }
};