#pragma once

#include<iostream>

#include "board.h"

/// @brief 
/// @param board 
/*
    piece: {}
    kings: ##

*/
void printBoard(const Board& board) {
    uint32_t blacks = board.black & ~board.kings;
    uint32_t whites = board.white & ~board.kings;

    for(int i = 0; i < 8; i++) {
        bool odd = i % 2;
        for(int j = 0; j < 4; j++) {
            int ind = i * 4 + j;
            if(!odd) printf("\033[48;5;255m  \033[0m");
            
            if (((board.black & board.kings) >> ind) & 1)
                printf("\033[30;40;1mBK\033[0m");            
            else if (((board.white & board.kings) >> ind) & 1)
                printf("\033[97;40;1mWK\033[0m");           
            else if((board.black >> ind) & 1)
                printf("\033[30;40;1mbb\033[0m");
            else if((board.white >> ind) & 1)
                printf("\033[97;40;1mww\033[0m");            
            else
                printf("\033[30;40m  \033[0m");
            
            if(odd) printf("\033[48;5;255m  \033[0m");
        }

        printf("\n");
    }
    printf("\n");
}

void printBoard(const Board& from, const Board& to) {

    // Squares that changed in any way (move, capture, kinging)
    uint32_t diff =
        (from.white ^ to.white) |
        (from.black ^ to.black) |
        (from.kings ^ to.kings);

    // Source squares (piece moved FROM here or was captured)
    uint32_t movedFrom =
        (from.white | from.black) & ~(to.white | to.black);

    // Destination squares (piece moved TO here)
    uint32_t movedTo =
        (to.white | to.black) & ~(from.white | from.black);

    for(int i = 0; i < 8; i++) {
        bool odd = i % 2;

        for(int j = 0; j < 4; j++) {
            int ind = i * 4 + j;

            if(!odd) printf("\033[48;5;255m  \033[0m");

            bool changed   = (diff >> ind) & 1;
            bool fromSq    = (movedFrom >> ind) & 1;
            bool toSq      = (movedTo >> ind) & 1;

            // Color priority:
            //   Green = destination
            //   Red   = source / captured
            //   Normal = unchanged
            const char* color =
                toSq   ? "\033[92;40;1m" :   // bright green
                fromSq ? "\033[91;40;1m" :   // bright red
                changed? "\033[93;40;1m" :   // yellow (kinging etc)
                         "\033[30;40;1m";   // normal black

            if (((to.black & to.kings) >> ind) & 1) {
                printf("%sBK\033[0m", color);
            } 
            else if (((to.white & to.kings) >> ind) & 1) {
                printf("%sWK\033[0m", color);
            } 
            else if ((to.black >> ind) & 1) {
                printf("%sbb\033[0m", color);
            } 
            else if ((to.white >> ind) & 1) {
                printf("%sww\033[0m", color);
            } 
            else {
                if (changed) printf("\033[93;40m  \033[0m");
                else         printf("\033[30;40m  \033[0m");
            }

            if(odd) printf("\033[48;5;255m  \033[0m");
        }

        printf("\n");
    }

    printf("\n");
}
