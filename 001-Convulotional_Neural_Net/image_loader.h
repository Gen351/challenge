#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

// --- NEW: Include the image decoder ---
// Define this macro ONLY in ONE .cpp file (usually main.cpp), 
// but for simple projects, defining it here with 'static' functions is okay 
// if you only include image_loader in main.
#define STB_IMAGE_IMPLEMENTATION
#include "cnn/stb_image.h" 
// --------------------------------------

#include "./cnn/matrix_op/matrix.hpp"
#include "./cnn/tensor.h"

namespace ImageLoader {

    struct Dataset {
        std::vector<Tensor> inputs;
        std::vector<Matrix<float>> targets;
    };

    class Image {
    public:
        int channels; 
        size_t rows, cols;
        std::vector<Matrix<float>> image2d; 

        Image(int c, int r, int col) 
            : channels(c), rows(r), cols(col) {
            for(int i = 0; i < c; i++) {
                image2d.push_back(Matrix<float>(r, col, 0.0f));
            }
        }

        Tensor toTensor() const {
            Tensor t;
            for(const auto& channelMat : image2d) {
                t.addFeatureMap(channelMat);
            }
            return t;
        }
    };

    // --- NEW: Single Image Loader ---
    // Usage: Tensor t = ImageLoader::loadImage("my_flower.jpg", 64, 64, 3);
    inline Tensor loadImage(const std::string& filepath, int targetW, int targetH, int targetChannels) {
        int width, height, channels;
        
        // Load image (forces 3 channels if targetChannels is 3, to handle RGBA or Grayscale conversion automatically)
        unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, targetChannels);

        if (!data) {
            throw std::runtime_error("Failed to load image: " + filepath + " (" + stbi_failure_reason() + ")");
        }

        // Create container for the resized result
        Image img(targetChannels, targetH, targetW);

        // Resize Loop (Nearest Neighbor) & Normalize
        // This converts the raw bytes (0-255) into your Model's 0.0-1.0 float format
        for (int y = 0; y < targetH; y++) {
            for (int x = 0; x < targetW; x++) {
                
                // Calculate corresponding pixel in source image
                int srcX = x * width / targetW;
                int srcY = y * height / targetH;

                // Index of the pixel in the 1D raw array
                // stbi loads as: [R, G, B, R, G, B, ...]
                int srcIndex = (srcY * width + srcX) * targetChannels;

                for (int c = 0; c < targetChannels; c++) {
                    unsigned char pixelVal = data[srcIndex + c];
                    img.image2d[c](y, x) = pixelVal / 255.0f;
                }
            }
        }

        // Free the raw memory loaded by stbi
        stbi_image_free(data);

        std::cout << "Loaded & Resized image: " << filepath << " [" << width << "x" << height << "] -> [" << targetW << "x" << targetH << "]" << std::endl;
        
        return img.toTensor();
    }

    // Main loading function (CSV)
    inline Dataset loadFromCSV(const std::string& filename, int height, int width, int channels, int numClasses) {
        Dataset dataset;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::getline(file, line); // Skip header

        int count = 0;
        std::cout << "Loading dataset: " << filename << "..." << std::endl;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string valStr;

            // 1. Read Label
            if (!std::getline(ss, valStr, ',')) continue;
            int label = std::stoi(valStr);

            Matrix<float> labelMat(1, numClasses, 0.0f);
            if(label >= 0 && label < numClasses) {
                labelMat(0, label) = 1.0f;
            }
            dataset.targets.push_back(labelMat);

            // 2. Parse Pixels
            Image img(channels, height, width);
            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++) {
                    for (int k = 0; k < channels; k++) {
                        if (std::getline(ss, valStr, ',')) {
                            float pixelVal = std::stof(valStr);
                            img.image2d[k](r, c) = pixelVal / 255.0f;
                        }
                    }
                }
            }
            dataset.inputs.push_back(img.toTensor());
            count++;
            
            if (count % 1000 == 0) std::cout << "Loaded " << count << " samples..." << std::endl;
        }

        std::cout << "Done! Total samples: " << dataset.inputs.size() << std::endl;
        return dataset;
    }
}