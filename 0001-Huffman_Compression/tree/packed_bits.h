#pragma once

#include "tree.h"
#include <cstdint>

// This represents the path to a character in the tree
struct BitCode {
    uint32_t bits = 0; // The actual bit pattern (e.g., 0b1011)
    uint8_t size = 0;  // Number of valid bits in the pattern
};

// This stores the actual compressed binary data
struct PackedBits {
    std::vector<uint8_t> data; // The raw compressed bytes
    size_t totalBits = 0;      // Needed for the decoder to know when to stop
};

struct BitsPacker {
    HuffmanTree tree;
    BitCode fastTable[256]; // Speed optimization: direct array lookup

    BitsPacker() {}

    // 1. Build the tree and the fast lookup table
    void init(const std::string& input) {
        tree.build(input);
        
        // We need a helper in your tree class to fill this BitCode table
        // instead of the std::string table. 
        generate_fast_table(tree.get_root(), 0, 0);
    }

    // 2. The high-performance packing loop
    void pack(const std::string& rawInput, PackedBits& packedBits) {
        if(tree.not_init()) return;

        uint64_t bitBuffer = 0; // Our "window" for bit manipulation
        int bitsInBuffer = 0;
        packedBits.data.clear();
        packedBits.totalBits = 0;

        for (unsigned char c : rawInput) {
            BitCode code = fastTable[c];
            
            // Append the new code to the right side of our buffer
            bitBuffer = (bitBuffer << code.size) | code.bits;
            bitsInBuffer += code.size;
            packedBits.totalBits += code.size;

            // Whenever we have 8 or more bits, "drain" them into a byte
            while (bitsInBuffer >= 8) {
                bitsInBuffer -= 8;
                // Shift bits right to grab the "oldest" 8 bits
                uint8_t byte = static_cast<uint8_t>(bitBuffer >> bitsInBuffer);
                packedBits.data.push_back(byte);
                
                // Clear the bits we just saved to the vector
                bitBuffer &= (1ULL << bitsInBuffer) - 1;
            }
        }

        // 3. Handle the "Trailing Bits"
        if (bitsInBuffer > 0) {
            // Shift left so the bits are "left-aligned" in the final byte
            uint8_t finalByte = static_cast<uint8_t>(bitBuffer << (8 - bitsInBuffer));
            packedBits.data.push_back(finalByte);
        }
    }

private:
    // Recursive DFS to populate the BitCode array
    void generate_fast_table(Node* node, uint32_t currentPath, uint8_t depth) {
        if (!node) return;

        // Leaf node: store the path
        if (!node->left && !node->right) {
            fastTable[(unsigned char)node->data] = { currentPath, depth };
            return;
        }

        // Left = 0, Right = 1
        generate_fast_table(node->left, (currentPath << 1), depth + 1);
        generate_fast_table(node->right, (currentPath << 1) | 1, depth + 1);
    }
};