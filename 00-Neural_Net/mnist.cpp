#include<iostream>
#include<fstream>

#include<vector>

#include "net.h"
#include "./dataset/network_formatter.h"
#include "./dataset/network_saver.h"
#include "./dataset/dataset_loader.h"

const std::string& MNIST_TRAIN_PATH = "./dataset/MINST/mnist_train.csv";

int main(int argc, char* argv[]) {

    std::vector<int> layerSizes = {784, 100, 50, 25, 10};
    Net cnn(layerSizes);

    cnn.DEBUG();

    return argc;
}