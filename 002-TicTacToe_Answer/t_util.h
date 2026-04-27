#pragma once

#include<vector>

enum class chip {
    o,
    x,
    i
};

enum state {
    not_done=0,
    x_win=-1,
    o_win=1,
    draw=2
};


namespace TicTac {
    
    /// @param board {chip:9}
    /// @return 0 if not done, -1 if x won, 1 if o won, 2 if draw
    int isDone(const std::vector<chip>& board) {
        // x check
        // horizontal
        if(board[0]==chip::x&&board[1]==chip::x&&board[2]==chip::x) return x_win;
        if(board[3]==chip::x&&board[4]==chip::x&&board[5]==chip::x) return x_win;
        if(board[6]==chip::x&&board[7]==chip::x&&board[8]==chip::x) return x_win;
        // vertical
        if(board[0]==chip::x&&board[3]==chip::x&&board[6]==chip::x) return x_win;
        if(board[1]==chip::x&&board[4]==chip::x&&board[7]==chip::x) return x_win;
        if(board[2]==chip::x&&board[5]==chip::x&&board[8]==chip::x) return x_win;
        // cross
        if(board[0]==chip::x&&board[4]==chip::x&&board[8]==chip::x) return x_win;
        if(board[2]==chip::x&&board[4]==chip::x&&board[6]==chip::x) return x_win;
        
        // o check
        // horizontal
        if(board[0]==chip::o&&board[1]==chip::o&&board[2]==chip::o) return o_win;
        if(board[3]==chip::o&&board[4]==chip::o&&board[5]==chip::o) return o_win;
        if(board[6]==chip::o&&board[7]==chip::o&&board[8]==chip::o) return o_win;
        // vertical
        if(board[0]==chip::o&&board[3]==chip::o&&board[6]==chip::o) return o_win;
        if(board[1]==chip::o&&board[4]==chip::o&&board[7]==chip::o) return o_win;
        if(board[2]==chip::o&&board[5]==chip::o&&board[8]==chip::o) return o_win;
        // cross
        if(board[0]==chip::o&&board[4]==chip::o&&board[8]==chip::o) return o_win;
        if(board[2]==chip::o&&board[4]==chip::o&&board[6]==chip::o) return o_win;

        // draw check
        for(const chip& chips : board) {
            if(chips == chip::i) return not_done;
        }

        return draw;
    }


    int eval(std::vector<chip>& board, chip turn, int depth) {
        int state = isDone(board);
        if(state == x_win) return -10 + depth;
        else if(state == o_win) return 10 - depth;
        else if(state == draw) return 0;

        int best = turn == chip::o ? -100000 : 100000;

        for(int i = 0; i < 9; i++) {
            if(board[i] == chip::i) {
                board[i] = turn;
                chip nextTurn = turn == chip::x ? chip::o : chip::x;
                int res = eval(board, nextTurn, depth + 1);
                if(turn == chip::o) {
                    if(res > best) {
                        best = res;
                    }
                } else {
                    if(res < best) {
                        best = res;
                    }
                }
                board[i] = chip::i;
            }
        }

        return best;
    }

    int answer(chip turn, std::vector<chip>& board) {
        if(isDone(board) == draw) return -1;

        int best_ind = 0;
        int best_score = turn == chip::o ? -100000 : 100000;
        for(int i = 0; i < 9; i++) {
            if(board[i] == chip::i) {
                board[i] = turn;
                chip nextTurn = turn == chip::x ? chip::o : chip::x;
                int res = eval(board, nextTurn, 0);
                if(turn == chip::o) {
                    if(res > best_score) {
                        best_score = res;
                        best_ind = i;
                    }
                } else {
                    if(res < best_score) {
                        best_score = res;
                        best_ind = i;
                    }
                }

                board[i] = chip::i;
            }
        }

        return best_ind;
    }
};