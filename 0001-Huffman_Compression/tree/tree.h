#pragma once

#include <vector>
#include<queue>
#include<string>

#include<iostream>

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

    bool operator<(const Node& other) {        
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
    std::vector<std::string> table;

public:
    HuffmanTree() : root(nullptr), table(256, "") {

    }
    HuffmanTree(const std::string& input) {
        build(input);
    }

    std::string encrypt(const std::string& input) {
        build(input);
        return code_encrypt(input);
    }

    std::string decrypt(const std::string& encrypted) {
        if(!root) return "?";

        return code_decrypt(encrypted);
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
        return root ? false : true;
    }

    /// @brief 
    /// @param mode 1:prefix 2:infix 3:postfix 
    void print_tree(int mode) {
        switch(mode) {
            case 1: print_tree_prefix(root); break;
            case 2: print_tree_infix(root); break;
            case 3: print_tree_postfix(root); break;
            default: break;
        }
    }

private:
    std::string code_decrypt(const std::string& bits) {
        if(root == nullptr) return "?";
        std::string decrypted = "";

        Node* traverse = root;
        
        for(int i = 0; i < bits.length(); i++) {
            if(!root->left || !root->right) {
                decrypted += root->data;
                continue;
            }

            // maybe just a guard for decrypting wrong encrypted data on the wrong tree
            if(traverse == nullptr) { // idk how it will get to this if the encrypt is correct.
                traverse = root;
            }
            
            if(bits[i] == '0') {
                traverse = traverse->left;
            } else if(bits[i] == '1'){
                traverse = traverse->right;
            }
            
            if(!traverse->left || !traverse->right) {
                decrypted += traverse->data;
                traverse = root;
            }
        }

        return decrypted;
    }

    std::string code_encrypt(const std::string& input) {
        if(input.empty()) return "?";
        std::string encrypted = "";
        for(int i = 0; i < input.size(); i++) {
            encrypted += table[(unsigned char)input[i]];
        }

        return encrypted;
    }

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

        set_table(root, "", table);
    }


    void set_table(Node* node, std::string code, std::vector<std::string>& table) {
        if(!node) return;

        if(!node->left || !node->right) {
            table[(unsigned char)node->data] = code;
            return;
        }

        set_table(node->left, code + '0', table);
        set_table(node->right, code + '1', table);
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


    int exist(const std::string& str, const char c) {
        if(str.empty()) return -1;

        for(int i = 0; i < str.length(); i++)
            if(str[i] == c)
                return i; 
        return -1;
    }

public:
    ~HuffmanTree() {
        clear_tree(root);
    }
};