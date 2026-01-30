#pragma once

#include "cnn/tensor.h"


namespace Visualize {

    void RGB(const Tensor& input) {
        if (input.featureMaps.empty()) return;

        // 1. Dimensions
        const size_t rows = input.featureMaps[0].rows();
        const size_t cols = input.featureMaps[0].cols();
        const size_t channels = input.featureMaps.size();
        const size_t totalPixels = rows * cols;

        // 2. Setup Pointers (Direct Memory Access)
        // FIX: Use .data.data() to get the raw pointer
        const float* pR = input.featureMaps[0].data.data();
        const float* pG = (channels > 1) ? input.featureMaps[1].data.data() : pR;
        const float* pB = (channels > 2) ? input.featureMaps[2].data.data() : pR;

        // 3. Prepare Buffer
        std::string frameBuffer;
        frameBuffer.reserve(totalPixels * 22 + rows); 
        
        char pxBuffer[32]; 

        // 4. Iterate (Row-Major)
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                
                float rVal = *pR++;
                float gVal = *pG++;
                float pVal = *pB++;

                // Fast clamp and convert
                int R = static_cast<int>(std::max(0.0f, std::min(1.0f, rVal)) * 255.99f);
                int G = static_cast<int>(std::max(0.0f, std::min(1.0f, gVal)) * 255.99f);
                int B = static_cast<int>(std::max(0.0f, std::min(1.0f, pVal)) * 255.99f);

                // Format pixel
                int len = snprintf(pxBuffer, sizeof(pxBuffer), "\x1b[48;2;%d;%d;%dm  ", R, G, B);
                frameBuffer.append(pxBuffer, len);
            }
            frameBuffer.append("\x1b[0m\n");
        }

        // 5. Output
        fwrite(frameBuffer.c_str(), 1, frameBuffer.size(), stdout);
        fflush(stdout); 
    }


    void featureMap(const Matrix<float>& map) {
        if (map.data.empty()) return;

        const size_t rows = map.rows();
        const size_t cols = map.cols();
        const size_t totalPixels = rows * cols;

        // --- STEP 1: Find Max for Normalization ---
        // Conv layers often output values > 1.0. Without this, 
        // your maps might just look like solid white blocks.
        float maxVal = 0.0001f; 
        for(float v : map.data) if(v > maxVal) maxVal = v;

        std::string frameBuffer;
        frameBuffer.reserve(totalPixels * 25 + rows); 
        char pxBuffer[64]; 

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                float val = map(r, c);
                
                // Normalize and clamp
                int intensity = static_cast<int>((val / maxVal) * 255.0f);
                if (intensity < 0) intensity = 0;
                if (intensity > 255) intensity = 255;

                // --- STEP 2: Use R;G;B triple for Grayscale ---
                // We pass 'intensity' three times to get gray.
                int len = snprintf(pxBuffer, sizeof(pxBuffer), 
                                "\x1b[48;2;%d;%d;%dm  ", 
                                intensity, intensity, intensity);
                frameBuffer.append(pxBuffer, len);
            }
            frameBuffer.append("\x1b[0m\n");
        }

        fwrite(frameBuffer.c_str(), 1, frameBuffer.size(), stdout);
        fflush(stdout);
    }



    std::string featureMap_string(const Matrix<float>& map) {
        if (map.data.empty()) return "";

        const size_t rows = map.rows();
        const size_t cols = map.cols();
        const size_t totalPixels = rows * cols;

        // --- STEP 1: Find Max for Normalization ---
        // Conv layers often output values > 1.0. Without this, 
        // your maps might just look like solid white blocks.
        float maxVal = 0.0001f; 
        for(float v : map.data) if(v > maxVal) maxVal = v;

        std::string frameBuffer;
        frameBuffer.reserve(totalPixels * 25 + rows); 
        char pxBuffer[64]; 

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                float val = map(r, c);
                
                // Normalize and clamp
                int intensity = static_cast<int>((val / maxVal) * 255.0f);
                if (intensity < 0) intensity = 0;
                if (intensity > 255) intensity = 255;

                // --- STEP 2: Use R;G;B triple for Grayscale ---
                // We pass 'intensity' three times to get gray.
                int len = snprintf(pxBuffer, sizeof(pxBuffer), 
                                "\x1b[48;2;%d;%d;%dm  ", 
                                intensity, intensity, intensity);
                frameBuffer.append(pxBuffer, len);
            }
            frameBuffer.append("\x1b[0m\n");
        }

        return frameBuffer;
    }

    std::string RGB_string(const Tensor& input) {
        if (input.featureMaps.empty()) return "";

        // 1. Dimensions
        const size_t rows = input.featureMaps[0].rows();
        const size_t cols = input.featureMaps[0].cols();
        const size_t channels = input.featureMaps.size();
        const size_t totalPixels = rows * cols;

        // 2. Setup Pointers (Direct Memory Access)
        // FIX: Use .data.data() to get the raw pointer
        const float* pR = input.featureMaps[0].data.data();
        const float* pG = (channels > 1) ? input.featureMaps[1].data.data() : pR;
        const float* pB = (channels > 2) ? input.featureMaps[2].data.data() : pR;

        // 3. Prepare Buffer
        std::string frameBuffer;
        frameBuffer.reserve(totalPixels * 22 + rows); 
        
        char pxBuffer[32]; 

        // 4. Iterate (Row-Major)
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                
                float rVal = *pR++;
                float gVal = *pG++;
                float pVal = *pB++;

                // Fast clamp and convert
                int R = static_cast<int>(std::max(0.0f, std::min(1.0f, rVal)) * 255.99f);
                int G = static_cast<int>(std::max(0.0f, std::min(1.0f, gVal)) * 255.99f);
                int B = static_cast<int>(std::max(0.0f, std::min(1.0f, pVal)) * 255.99f);

                // Format pixel
                int len = snprintf(pxBuffer, sizeof(pxBuffer), "\x1b[48;2;%d;%d;%dm  ", R, G, B);
                frameBuffer.append(pxBuffer, len);
            }
            frameBuffer.append("\x1b[0m\n");
        }

        return frameBuffer;
    }

}