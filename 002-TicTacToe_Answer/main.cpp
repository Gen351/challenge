#include<iostream>

#include "t_util.h"

#define o chip::o
#define x chip::x
#define i chip::i

int main() {
    std::vector<chip> board(0, i);
    
    board = {
        o, o, x,
        x, i, o,
        o, o, x
    };

    std::cout << TicTac::answer(x, board);

    return 0;
}