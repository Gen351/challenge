#include<iostream>

#include<time.h>
#include<random>

#include "layer.h"
#include "net.h"

int main () {
    srand(time(0));

    // Test 3
    Layer layer = Layer();
    layer.DEBUG("LAYER DEFAULT TEST");

    // Test 2
    Layer layer1 = Layer(2, 4);
    layer1.DEBUG("LAYER TEST(2, 4)");
    Layer layer2 = Layer(4, 2);
    layer2.DEBUG("LAYER TEST(4, 2)");

    // Test 3
    Net net = Net();
    net.DEBUG("NET DEFAULT TEST");

    //Test 4
    std::vector<int> layerSizes = {2, 4, 4, 2, 2};
    Net net1 = Net(layerSizes);
    net1.DEBUG("NET TEST{2, 4, 4, 2, 2}");

    return 0;
}

