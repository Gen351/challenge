#include <iostream>
#include <string>
#include <iomanip>
#include "tree/efficient_packed_bits.h"
#include <fstream>
#include <iterator>

std::string get(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return "";
    }

    // Efficiently read the entire file into a string
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

int main(int argc, char* argv[]) {
    std::cout << " --Reading file into std::string\n";
    std::string original = get("importantDoc2.txt");

    // 1. Setup the Packer and Tree
    BitsPacker packer;
    std::cout << " --Making the compresor...\n";
    packer.init(original); // Builds tree and fastTable internally
    
    // 2. Compress (Pack)
    PackedBits compressed;
    std::cout << " --Packing...\n";
    packer.pack(original, compressed);

    std::cout << "Original Length: " << " (" << original.length() * 8 << " bits) | " << "(" << (double)original.length() / (1024 * 1024) << " MB)\n";
    std::cout << "Compressed size: " << compressed.data.size() * 8 << " bits (including padding) | " << "(" << (double)compressed.data.size() / (1024 * 1024) << " MB)\n";
    std::cout << "Actual useful bits: " << compressed.totalBits << "\n";
    
    std::cout << "Compressed to " << (((double)compressed.data.size()) / ((double)original.length())) * 100 << "% of Original Size\n\n";

    // 3. Decompress (Unpack)
    // NOTE: Building a std::string of '0's and '1's was removed here. 
    // It causes a massive memory leak (8x the file size).
    // Instead, we pass the raw bytes directly to the tree to decode on the fly!

    // 4. Final Decryption
    std::cout << " --Unpacking...\n";
    std::string decoded = packer.tree.decrypt(compressed.data, compressed.totalBits);

    std::cout << " --Checking Integrity...\n";
    if (original == decoded) {
        std::cout << "SUCCESS: Data integrity verified.\n";
    } else {
        std::cout << "DECODING ERROR: Data integrity not verified.\n";
    }

    return 0;
}