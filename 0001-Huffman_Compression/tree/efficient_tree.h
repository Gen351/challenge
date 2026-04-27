#pragma once

#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <cstdint>

struct Node {
    int frequency;
    char data;

    Node* left;
    Node* right;
    
    Node(int f, char d) 
        : frequency(f), data(d) {
            left = right = nullptr;
    }

    Node(int f) 
        : frequency(f) {
            left = right = nullptr;
            data = 127;
    }

    bool operator<(const Node& other) const {        
        return frequency > other.frequency;
    }
};

struct CompareNode {
    bool operator()(Node* const& n1, Node* const& n2) {
        return n1->frequency > n2->frequency;
    }
};

class HuffmanTree {
    Node* root;

public:
    HuffmanTree() : root(nullptr) {}
    
    HuffmanTree(const std::string& input) {
        build(input);
    }

    void build(const std::string& data) {
        if(data.empty()) {
            std::cerr << "Empty Data\n";
            return;
        }

        std::vector<Node> nodes;
        int frequencies[256] = {0};

        for(int i = 0; i < data.length(); i++)
            frequencies[(unsigned char)data[i]]++;

        for(int i = 0; i < 256; i++)
            if(frequencies[i] > 0) 
                nodes.emplace_back(frequencies[i], (char)i);

        build_tree(nodes);
    }

    Node* get_root() {
        return root;
    }

    bool not_init() {
        return root == nullptr;
    }

    // MEMORY EFFICIENT DECRYPT: Reads bits directly from the packed byte array
    std::string decrypt(const std::vector<uint8_t>& compressedData, size_t totalBits) {
        if(!root) return "?";
        std::string decrypted = "";
        
        // Pre-allocate memory for the decrypted string to prevent reallocation overhead
        // Since we don't know the exact length, we'll let it grow naturally, 
        // but it skips the massive 8x memory bloat of the "0101" string.

        Node* traverse = root;
        size_t bitsRead = 0;

        // Edge Case: Tree only has one node (e.g., input was "AAAA")
        if (!root->left && !root->right) {
            return std::string(totalBits, root->data);
        }

        for (uint8_t byte : compressedData) {
            for (int i = 7; i >= 0; i--) {
                if (bitsRead >= totalBits) break; // Stop reading padding bits

                if ((byte >> i) & 1) {
                    traverse = traverse->right;
                } else {
                    traverse = traverse->left;
                }
                bitsRead++;

                // If we hit a leaf, append the character and reset to root
                if (!traverse->left && !traverse->right) {
                    decrypted += traverse->data;
                    traverse = root;
                }
            }
        }

        return decrypted;
    }

    void print_tree(int mode) {
        switch(mode) {
            case 1: print_tree_prefix(root); break;
            case 2: print_tree_infix(root); break;
            case 3: print_tree_postfix(root); break;
            default: break;
        }
    }

    ~HuffmanTree() {
        clear_tree(root);
    }

private:
    void build_tree(const std::vector<Node>& nodes) {
        std::priority_queue<Node*, std::vector<Node*>, CompareNode> nodeQueue;
        
        for(const auto& node : nodes) {
            Node* nodePtr = new Node(node.frequency, node.data);
            nodeQueue.push(nodePtr);
        }

        while(nodeQueue.size() > 1) {
            Node* nodeLeft = nodeQueue.top();
            nodeQueue.pop();
            
            Node* nodeRight = nodeQueue.top();
            nodeQueue.pop();
            
            int freqSum = nodeLeft->frequency + nodeRight->frequency;
            Node* newNode = new Node(freqSum);
            newNode->left = nodeLeft;
            newNode->right = nodeRight;
            
            nodeQueue.push(newNode);
        }

        if(!nodeQueue.empty()) {
            root = nodeQueue.top();
            nodeQueue.pop();
        }
    }

    void print_tree_postfix(Node* node) {
        if(node) {
            print_tree_postfix(node->left);
            print_tree_postfix(node->right);
            std::cout << node->data << ": " << node->frequency << "\n";
        }
    }
    void print_tree_prefix(Node* node) {
        if(node) {
            std::cout << node->data << ": " << node->frequency << "\n";
            print_tree_prefix(node->left);
            print_tree_prefix(node->right);
        }
    }

    void print_tree_infix(Node* node) {
        if(node) {
            print_tree_infix(node->left);
            std::cout << node->data << ": " << node->frequency << "\n";
            print_tree_infix(node->right);
        }
    }

    void clear_tree(Node* node) {
        if(node) {
            clear_tree(node->left);
            clear_tree(node->right);
            delete node;
        }
    }
};