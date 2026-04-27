#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>

using namespace std;

// ------------------------------------------------------------
// Huffman tree node (used during compression)
// ------------------------------------------------------------
struct Node {
    int freq;
    unsigned char byte;       // meaningful only for leaves
    Node *left, *right;

    Node(int f, unsigned char b, Node* l = nullptr, Node* r = nullptr)
        : freq(f), byte(b), left(l), right(r) {}
};

// Min‑heap comparator for Node pointers
struct Compare {
    bool operator()(Node* a, Node* b) const {
        return a->freq > b->freq;
    }
};

// Recursively delete a Huffman tree
void deleteTree(Node* root) {
    if (!root) return;
    deleteTree(root->left);
    deleteTree(root->right);
    delete root;
}

// ------------------------------------------------------------
// Generate code lengths from the Huffman tree
// ------------------------------------------------------------
void generateLengths(Node* root, vector<int>& lengths, int depth = 0) {
    if (!root) return;
    if (!root->left && !root->right) {          // leaf
        lengths[root->byte] = depth;
    } else {
        generateLengths(root->left, lengths, depth + 1);
        generateLengths(root->right, lengths, depth + 1);
    }
}

// ------------------------------------------------------------
// Build canonical Huffman codes from the given lengths
// ------------------------------------------------------------
void buildCanonicalCodes(const vector<int>& lengths, vector<string>& codes) {
    int maxLen = 0;
    for (int len : lengths)
        if (len > maxLen) maxLen = len;

    vector<int> count(maxLen + 1, 0);
    for (int len : lengths)
        if (len > 0) count[len]++;

    vector<int> firstCode(maxLen + 1, 0);
    int code = 0;
    for (int len = 1; len <= maxLen; ++len) {
        code = (code + count[len - 1]) << 1;
        firstCode[len] = code;
    }

    codes.assign(256, "");
    for (int sym = 0; sym < 256; ++sym) {
        int len = lengths[sym];
        if (len > 0) {
            int codeVal = firstCode[len]++;
            string bits(len, '0');
            for (int i = len - 1; i >= 0; --i) {
                bits[i] = (codeVal & 1) ? '1' : '0';
                codeVal >>= 1;
            }
            codes[sym] = bits;
        }
    }
}

// ------------------------------------------------------------
// BitWriter – packs bits into bytes and writes to an ofstream
// ------------------------------------------------------------
class BitWriter {
    ofstream& out;
    unsigned char buffer;
    int bitsInBuffer;

public:
    BitWriter(ofstream& os) : out(os), buffer(0), bitsInBuffer(0) {}
    ~BitWriter() { flush(); }

    void writeBit(int bit) {
        buffer = (buffer << 1) | (bit & 1);
        ++bitsInBuffer;
        if (bitsInBuffer == 8) {
            out.write(reinterpret_cast<char*>(&buffer), 1);
            bitsInBuffer = 0;
            buffer = 0;
        }
    }

    void flush() {
        if (bitsInBuffer > 0) {
            buffer <<= (8 - bitsInBuffer);   // pad with zeros on the right
            out.write(reinterpret_cast<char*>(&buffer), 1);
            bitsInBuffer = 0;
            buffer = 0;
        }
    }
};

// ------------------------------------------------------------
// BitReader – reads bits from an ifstream (MSB first)
// ------------------------------------------------------------
class BitReader {
    ifstream& in;
    unsigned char buffer;
    int bitsLeft;

public:
    BitReader(ifstream& is) : in(is), buffer(0), bitsLeft(0) {}

    // Returns -1 on EOF, otherwise 0 or 1
    int readBit() {
        if (bitsLeft == 0) {
            if (!in.read(reinterpret_cast<char*>(&buffer), 1))
                return -1;
            bitsLeft = 8;
        }
        int bit = (buffer >> 7) & 1;   // take the most significant bit
        buffer <<= 1;
        --bitsLeft;
        return bit;
    }
};

// ------------------------------------------------------------
// Compression routine
// ------------------------------------------------------------
void compress(const string& inputPath) {
    ifstream inFile(inputPath, ios::binary);
    if (!inFile) {
        cerr << "Error: Cannot open input file " << inputPath << endl;
        return;
    }

    // First pass: count frequencies
    vector<int> freq(256, 0);
    unsigned char byte;
    uint64_t totalBytes = 0;
    while (inFile.read(reinterpret_cast<char*>(&byte), 1)) {
        ++freq[byte];
        ++totalBytes;
    }

    // Build Huffman tree
    priority_queue<Node*, vector<Node*>, Compare> pq;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0)
            pq.push(new Node(freq[i], static_cast<unsigned char>(i)));
    }

    Node* root = nullptr;
    if (pq.empty()) {
        root = nullptr;                     // empty file
    } else {
        while (pq.size() > 1) {
            Node* left = pq.top(); pq.pop();
            Node* right = pq.top(); pq.pop();
            Node* parent = new Node(left->freq + right->freq, 0, left, right);
            pq.push(parent);
        }
        root = pq.top(); pq.pop();
    }

    // Generate code lengths
    vector<int> lengths(256, 0);
    if (root)
        generateLengths(root, lengths);

    // Build canonical codes
    vector<string> codes(256);
    if (root)
        buildCanonicalCodes(lengths, codes);

    // Write compressed file
    string outPath = "compressed_" + inputPath;
    ofstream outFile(outPath, ios::binary);
    if (!outFile) {
        cerr << "Error: Cannot create output file " << outPath << endl;
        deleteTree(root);
        return;
    }

    // Header: magic "HUF1"
    const char magic[] = "HUF1";
    outFile.write(magic, 4);

    // Original size (uint64_t)
    outFile.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));

    // Code lengths (256 bytes)
    for (int i = 0; i < 256; ++i) {
        unsigned char len = static_cast<unsigned char>(lengths[i]);
        outFile.write(reinterpret_cast<const char*>(&len), 1);
    }

    // Encode data (if any)
    if (totalBytes > 0) {
        inFile.clear();
        inFile.seekg(0, ios::beg);          // rewind for second pass

        BitWriter bitWriter(outFile);
        while (inFile.read(reinterpret_cast<char*>(&byte), 1)) {
            const string& code = codes[byte];
            for (char bit : code)
                bitWriter.writeBit(bit == '1' ? 1 : 0);
        }
        bitWriter.flush();
    }

    inFile.close();
    outFile.close();
    deleteTree(root);

    cout << "Compressed to " << outPath << endl;
}

// ------------------------------------------------------------
// Decompression routine
// ------------------------------------------------------------
void decompress(const string& inputPath) {
    ifstream inFile(inputPath, ios::binary);
    if (!inFile) {
        cerr << "Error: Cannot open input file " << inputPath << endl;
        return;
    }

    // Read and verify header
    char magic[4];
    inFile.read(magic, 4);
    if (inFile.gcount() != 4 || strncmp(magic, "HUF1", 4) != 0) {
        cerr << "Error: File not compressed with this program or corrupted (invalid magic)" << endl;
        return;
    }

    uint64_t originalSize;
    inFile.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
    if (!inFile) {
        cerr << "Error: Failed to read original size" << endl;
        return;
    }

    vector<int> lengths(256);
    for (int i = 0; i < 256; ++i) {
        unsigned char len;
        inFile.read(reinterpret_cast<char*>(&len), 1);
        if (!inFile) {
            cerr << "Error: Failed to read code lengths" << endl;
            return;
        }
        lengths[i] = len;
    }

    // Reconstruct canonical codes (same as in compression)
    vector<string> codes(256);
    buildCanonicalCodes(lengths, codes);

    // Build decoding tree (binary trie)
    struct DecodeNode {
        int symbol;               // -1 for internal nodes
        DecodeNode *left, *right;
        DecodeNode() : symbol(-1), left(nullptr), right(nullptr) {}
    };

    DecodeNode* root = new DecodeNode();
    for (int sym = 0; sym < 256; ++sym) {
        if (lengths[sym] > 0) {
            const string& code = codes[sym];
            DecodeNode* curr = root;
            for (char bit : code) {
                if (bit == '0') {
                    if (!curr->left) curr->left = new DecodeNode();
                    curr = curr->left;
                } else {
                    if (!curr->right) curr->right = new DecodeNode();
                    curr = curr->right;
                }
            }
            curr->symbol = sym;
        }
    }

    // Handle empty file
    if (originalSize == 0) {
        string outPath = "unpacked_" + inputPath;
        ofstream outFile(outPath, ios::binary);
        outFile.close();
        cout << "Decompressed to " << outPath << " (empty file)" << endl;

        // Cleanup tree
        function<void(DecodeNode*)> deleteNode = [&](DecodeNode* node) {
            if (!node) return;
            deleteNode(node->left);
            deleteNode(node->right);
            delete node;
        };
        deleteNode(root);
        return;
    }

    // Prepare output file
    string outPath = "unpacked_" + inputPath;
    ofstream outFile(outPath, ios::binary);
    if (!outFile) {
        cerr << "Error: Cannot create output file " << outPath << endl;
        return;
    }

    BitReader bitReader(inFile);
    uint64_t bytesDecoded = 0;
    DecodeNode* curr = root;

    // Special case: tree is a single leaf (only one symbol)
    if (root->left == nullptr && root->right == nullptr && root->symbol != -1) {
        unsigned char sym = static_cast<unsigned char>(root->symbol);
        for (uint64_t i = 0; i < originalSize; ++i)
            outFile.write(reinterpret_cast<const char*>(&sym), 1);
    } else {
        while (bytesDecoded < originalSize) {
            int bit = bitReader.readBit();
            if (bit == -1) {
                cerr << "Error: Unexpected EOF during decoding" << endl;
                break;
            }
            curr = (bit == 0) ? curr->left : curr->right;
            if (!curr) {
                cerr << "Error: Invalid bit stream" << endl;
                break;
            }
            if (curr->symbol != -1) {      // reached a leaf
                unsigned char sym = static_cast<unsigned char>(curr->symbol);
                outFile.write(reinterpret_cast<const char*>(&sym), 1);
                ++bytesDecoded;
                curr = root;                // restart at root
            }
        }
    }

    outFile.close();
    inFile.close();

    // Cleanup tree
    function<void(DecodeNode*)> deleteNode = [&](DecodeNode* node) {
        if (!node) return;
        deleteNode(node->left);
        deleteNode(node->right);
        delete node;
    };
    deleteNode(root);

    cout << "Decompressed to " << outPath << endl;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <filepath> <comp or unpack>" << endl;
        return 1;
    }

    string filepath = argv[1];
    string mode = argv[2];

    if (mode == "comp") {
        compress(filepath);
    } else if (mode == "unpack") {
        decompress(filepath);
    } else {
        cerr << "Invalid mode. Use 'comp' or 'unpack'." << endl;
        return 1;
    }

    return 0;
}