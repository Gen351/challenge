#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
#include "board.h"

// --- CONSTANTS ---
const int INF = 1000000;
const int CHECKER_VAL = 100;
const int KING_VAL = 300;
const int CENTER_BONUS = 15; 
const int BACK_RANK_BONUS = 10; 

// Global random engine (seeded once)
static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

int evaluate(const Board& b) {
    int whiteScore = 0;
    int blackScore = 0;

    // Evaluate White (Moves 31 -> 0, wants Row 0)
    for (int i=0; i<32; i++) {
        if ((b.white >> i) & 1) {
            int val = CHECKER_VAL;
            if ((b.kings >> i) & 1) val = KING_VAL;
            
            int row = i / 4;
            
            // Positional: Reward getting closer to Row 0
            // Row 7 (Start) = 0 pts, Row 0 (King) = 14 pts
            val += (7 - row) * 2; 

            // Center Control (14,15,18,19)
            if (i == 14 || i == 15 || i == 18 || i == 19) val += CENTER_BONUS;

            // Back Rank Safety (Row 7 for White's home? No, White starts at 20-31 (Rows 5-7))
            // Keeping pieces on Row 7 prevents Black from Kinging.
            if (row == 7) val += BACK_RANK_BONUS;

            whiteScore += val;
        }
    }

    // Evaluate Black (Moves 0 -> 31, wants Row 7)
    for (int i=0; i<32; i++) {
        if ((b.black >> i) & 1) {
            int val = CHECKER_VAL;
            if ((b.kings >> i) & 1) val = KING_VAL;
            
            int row = i / 4;
            
            // Positional: Reward getting closer to Row 7
            // Row 0 (Start) = 0 pts, Row 7 (King) = 14 pts
            val += row * 2;

            // Center Control
            if (i == 14 || i == 15 || i == 18 || i == 19) val += CENTER_BONUS;

            // Back Rank Safety (Row 0 is Black's home)
            if (row == 0) val += BACK_RANK_BONUS;

            blackScore += val;
        }
    }

    // Negamax Metric: (My Score - Enemy Score)
    if (b.whiteToMove) return whiteScore - blackScore; 
    return blackScore - whiteScore;
}

struct MoveSorter {
    const Board& original;
    MoveSorter(const Board& b) : original(b) {}
    
    bool operator()(const Board& a, const Board& b) {
        int countA = __builtin_popcount(a.white | a.black);
        int countB = __builtin_popcount(b.white | b.black);
        int kA = __builtin_popcount(a.kings);
        int kB = __builtin_popcount(b.kings);

        // 1. Prioritize Captures (Fewer pieces = capture happened)
        if (countA != countB) return countA < countB; 
        // 2. Prioritize Kings
        return kA > kB; 
    }
};

int negamax(Board b, int depth, int alpha, int beta) {
    if (depth == 0) return evaluate(b);

    std::vector<Board> states = b.getPossibleStates();

    // Game Over / No Moves
    if (states.empty()) return -INF + 100; 

    // Move Ordering to help Pruning
    std::sort(states.begin(), states.end(), MoveSorter(b));

    int bestVal = -INF;

    for (const Board& nextState : states) {
        int val = -negamax(nextState, depth - 1, -beta, -alpha);

        if (val > bestVal) {
            bestVal = val;
        }
        if (bestVal > alpha) {
            alpha = bestVal;
        }
        if (alpha >= beta) {
            break; 
        }
    }
    return bestVal;
}

Board getBestMove(Board b, int depth) {
    std::vector<Board> states = b.getPossibleStates();
    if(states.empty()) return b;

    // Sort to ensure we look at good moves first (efficiency)
    std::sort(states.begin(), states.end(), MoveSorter(b));

    // We will store ALL moves that share the highest score here
    std::vector<Board> bestMoves;
    int bestScore = -INF;

    for (const Board& state : states) {
        int score = -negamax(state, depth - 1, -INF, INF);
        
        if (score > bestScore) {
            bestScore = score;
            bestMoves.clear();
            bestMoves.push_back(state);
        } else if (score == bestScore) {
            bestMoves.push_back(state);
        }
    }
    
    // RANDOM TIE-BREAKER
    // This prevents the AI from playing the exact same game every time
    if (!bestMoves.empty()) {
        std::uniform_int_distribution<int> dist(0, bestMoves.size() - 1);
        return bestMoves[dist(rng)];
    }

    return states[0]; // Fallback
}