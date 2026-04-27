#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>

// --- CORRECTED NEIGHBOR TABLE ---
// Directions: 0=UL, 1=UR, 2=DL, 3=DR
// -1 means "Wall" or "Off Board"
static const int NEIGHBORS[32][4] = {
    // Row 0 (Even): 0,1,2,3
    {-1, -1, 4, 5},   {-1, -1, 5, 6},   {-1, -1, 6, 7},   {-1, -1, 7, -1},
    // Row 1 (Odd): 4,5,6,7
    {-1, 0, -1, 8},   {0, 1, 8, 9},     {1, 2, 9, 10},    {2, 3, 10, 11},
    // Row 2 (Even): 8,9,10,11
    {4, 5, 12, 13},   {5, 6, 13, 14},   {6, 7, 14, 15},   {7, -1, 15, -1},
    // Row 3 (Odd): 12,13,14,15
    {-1, 8, -1, 16},  {8, 9, 16, 17},   {9, 10, 17, 18},  {10, 11, 18, 19},
    // Row 4 (Even): 16,17,18,19
    {12, 13, 20, 21}, {13, 14, 21, 22}, {14, 15, 22, 23}, {15, -1, 23, -1},
    // Row 5 (Odd): 20,21,22,23
    {-1, 16, -1, 24}, {16, 17, 24, 25}, {17, 18, 25, 26}, {18, 19, 26, 27},
    // Row 6 (Even): 24,25,26,27
    {20, 21, 28, 29}, {21, 22, 29, 30}, {22, 23, 30, 31}, {23, -1, 31, -1},
    // Row 7 (Odd): 28,29,30,31
    {-1, 24, -1, -1}, {24, 25, -1, -1}, {25, 26, -1, -1}, {26, 27, -1, -1}
};

struct Board {
    uint32_t black;
    uint32_t white;
    uint32_t kings;
    bool whiteToMove;

    Board(uint32_t black=0x00000FFF, uint32_t white=0xFFF00000, uint32_t kings=0, bool whiteToMove=false)
        : black(black), white(white), kings(kings), whiteToMove(whiteToMove) {}

    // Check if a specific square has ANY piece
    bool isOccupied(int sq) const {
        return ((black | white) >> sq) & 1;
    }

    void removePiece(uint32_t sq) {
        uint32_t mask = ~(1U << sq);
        black &= mask; white &= mask; kings &= mask;
    }

    // --- SAFE MOVE APPLIER ---
    // Now includes a safety check to prevent "eating" on slides
    void applyMove(uint32_t from, uint32_t to) {
        // SAFETY: If we are moving to an occupied square, something is wrong.
        // We only overwrite 'to' if it's currently empty.
        // (Captures are handled by removePiece BEFORE calling applyMove)
        if (isOccupied(to)) {
             // In a real engine, we might assert(false) here. 
             // For now, we return to prevent the bug.
             return; 
        }

        uint32_t fromMask = (1U << from);
        uint32_t toMask = (1U << to);

        if (whiteToMove) { white &= ~fromMask; white |= toMask; } 
        else             { black &= ~fromMask; black |= toMask; }

        if (kings & fromMask) { kings &= ~fromMask; kings |= toMask; }

        // Promotion
        if (whiteToMove && to < 4) kings |= toMask;
        if (!whiteToMove && to > 27) kings |= toMask;
    }

    // --- RECURSIVE JUMP GENERATOR ---
    void findJumpChains(uint32_t currentSq, Board currentBoard, std::vector<Board>& moves) {
        bool isKing = (currentBoard.kings >> currentSq) & 1;
        bool foundContinuation = false;

        for (int dir = 0; dir < 4; dir++) {
            // Direction Constraints (Men only move forward)
            if (!isKing) {
                if (currentBoard.whiteToMove && (dir == 2 || dir == 3)) continue;
                if (!currentBoard.whiteToMove && (dir == 0 || dir == 1)) continue;
            }

            int mid = NEIGHBORS[currentSq][dir];
            if (mid == -1) continue;

            int dest = NEIGHBORS[mid][dir]; // Jump is 2 steps in same dir
            if (dest == -1) continue;

            // Valid Jump Logic:
            // 1. Dest is EMPTY
            // 2. Mid is ENEMY
            uint32_t enemies = currentBoard.whiteToMove ? currentBoard.black : currentBoard.white;
            
            if (!currentBoard.isOccupied(dest) && ((enemies >> mid) & 1)) {
                
                Board next = currentBoard;
                next.removePiece(mid);       // Eat the enemy
                next.applyMove(currentSq, dest); // Move self

                bool wasKing = isKing;
                bool nowKing = (next.kings >> dest) & 1;
                
                // If we just promoted, turn ends.
                if (!wasKing && nowKing) {
                    next.whiteToMove = !next.whiteToMove;
                    moves.push_back(next);
                    foundContinuation = true;
                } else {
                    // Recurse: Can we double jump?
                    size_t startSize = moves.size();
                    findJumpChains(dest, next, moves);
                    
                    if (moves.size() > startSize) {
                        foundContinuation = true;
                    } else {
                        // No more jumps, finalize this move
                        next.whiteToMove = !next.whiteToMove;
                        moves.push_back(next);
                        foundContinuation = true;
                    }
                }
            }
        }
    }

    std::vector<Board> getPossibleStates() {
        std::vector<Board> states;
        states.reserve(16);
        uint32_t myPieces = whiteToMove ? white : black;

        // 1. JUMPS (Mandatory)
        for (int i = 0; i < 32; i++) {
            if ((myPieces >> i) & 1) {
                findJumpChains(i, *this, states);
            }
        }
        if (!states.empty()) return states;

        // 2. SLIDES (Only if no jumps)
        for (int i = 0; i < 32; i++) {
            if (!((myPieces >> i) & 1)) continue;
            bool isKing = (kings >> i) & 1;

            for (int dir = 0; dir < 4; dir++) {
                if (!isKing) {
                    if (whiteToMove && (dir == 2 || dir == 3)) continue;
                    if (!whiteToMove && (dir == 0 || dir == 1)) continue;
                }

                int dest = NEIGHBORS[i][dir];
                if (dest == -1) continue;

                // STRICT CHECK: Destination must be empty
                if (!isOccupied(dest)) {
                    Board next = *this;
                    next.applyMove(i, dest);
                    next.whiteToMove = !whiteToMove;
                    states.push_back(next);
                }
            }
        }
        return states;
    }
};