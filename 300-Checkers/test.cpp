#include <iostream>
#include <vector>
#include <windows.h> // For Sleep and cls

#include "board.h"
#include "visualizer.h" // Assumes you have your printBoard here
#include "minimax.h"

// Helper to compare two boards manually
bool isSameState(const Board& a, const Board& b) {
    return (a.black == b.black) && 
           (a.white == b.white) && 
           (a.kings == b.kings) &&
           (a.whiteToMove == b.whiteToMove);
}

int main(int argc, char* argv[]) {
    // Initialize Board
    Board b = Board();

    // b.makeMove(11, 19, true);

    // History of board states to detect repetition
    std::vector<Board> history;
    int bounceCounter = 0;

    // Add initial state
    history.push_back(b);

    int moves = 0;
    while (true) {
        // 1. Generate the best move
        // Depth 8 is strong/fast with the new optimized Alpha-Beta
        Board next = getBestMove(b, moves > 50 ? 4 : 10); 
        moves++;

        // 2. Check for Game Over (If AI returns current state, it has no moves)
        if (isSameState(b, next)) {
            system("cls");
            printBoard(b, next); // Show final position
            std::cout << "Score: " << evaluate(b) << "\n";
            std::cout << "\nGame Over! " << (b.whiteToMove ? "Black" : "White") << " Wins (No moves left)!\n";
            break;
        }

        // 3. Visualization
        // Move cursor to top-left to reduce flickering (instead of constant cls)
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        COORD coord = {0, 0};
        SetConsoleCursorPosition(hOut, coord);
        
        // If the console is not clearing correctly, uncomment the next line:
        // system("cls"); 

        std::cout << "Turn: " << (b.whiteToMove ? "White" : "Black") << " | Score: " << evaluate(b) << "   \n";
        printBoard(b, next);

        // 4. Update Board State
        // (The new board.h ALREADY flips whiteToMove in 'next', so we just assign)
        b = next;
        
        // 5. Detect Bouncing (Repetition)
        // We check if the NEW state 'b' is identical to the state 2 moves ago (history[size - 2])
        // Sequence: A -> B -> A (This is 1 bounce)
        if (history.size() >= 2) {
            Board twoMovesAgo = history[history.size() - 2];
            
            if (isSameState(b, twoMovesAgo)) {
                bounceCounter++;
                std::cout << "\n[!] Repetition Detected: " << bounceCounter << "/5\n";
            } else {
                bounceCounter = 0;
            }
        }

        if (bounceCounter >= 5) {
            std::cout << "\n-----------------------------------\n";
            std::cout << " DRAW: States bounced 5 times.\n";
            std::cout << "-----------------------------------\n";
            break;
        }

        // Save state to history
        history.push_back(b);

        // Limit history size to prevent memory waste (we only need the last few)
        if (history.size() > 10) {
            history.erase(history.begin());
        }

        Sleep(10); // Speed up slightly (50 -> 20)
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}