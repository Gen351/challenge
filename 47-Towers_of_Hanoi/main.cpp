#include<iostream>
#include<vector>
#include<stdexcept>
#include<conio.h>
#include<windows.h>

const int TOWER_COUNT = 3;
const int TOWER_WIDTH = 17;

const char* disc_visual[] = {"{X}", 
                             "{XOX}",
                             "{TOWER}",
                             "{OFHANOI}",
                             "{ALEXANDRA}"}; 
const char* disc_pads[] = {"       ",
                           "      ",
                           "     ",
                           "    ",
                           "   "};
const char* disc_blank_padding = "                 ";

const char* empty = "|";
const char* empty_pads = "        ";

const std::string valid_char_inputs = {"ad p"};

enum op {
    take_disc,
    place_disc,
    move_left,
    move_right,
    stop,
    invalid_input
};

enum class disc_status {
    on_tower,
    hand,
    invalid
};

struct disc {
    // weight: (0) ~> (n-1)
    int tower_id;
    int weight;
    disc_status status;

    disc() {
        tower_id = -1;
        status = disc_status::invalid;
        weight = -1;
    }
    disc(disc_status stat, int w) {
        status = stat;
        weight = w;
        tower_id = -1;
    }
    disc(int w) {
        status = disc_status::invalid;
        weight = w;
        tower_id = -1;
    }
    disc& operator=(const disc& other) {
        if(this == &other) return *this;
        
        tower_id = other.tower_id;
        weight = other.weight;
        status = other.status;
        
        return *this;
    }
};

struct tower {
    int id;
    int capacity;
    int disc_count = 0;
    disc* discs;


    tower(int i, int cap) {
        id = i;
        printf("Init tower: %d..\n", id);

        capacity = cap;
        discs = new disc[capacity];

        if(i == 0) {
            disc_count = cap;
            for(int w = cap - 1; w >= 0; w--) {
                printf("T:%d|[D:w:%d]\t", id, w);
                discs[disc_count - w - 1] = disc(disc_status::on_tower, w);
                discs[disc_count - w - 1].tower_id = i;
                discs[disc_count - w - 1].status = disc_status::on_tower;
            }
        }

        printf("\nTower: %d|[D:%d] done.\n", id, disc_count);
    }

    tower(const tower& other) {
        id = other.id;
        capacity = other.capacity;
        disc_count = other.disc_count;
        
        discs = new disc[capacity]; // Allocate new memory
        for (int i = 0; i < disc_count; i++) {
            discs[i] = other.discs[i];
        }
    }

    void push(disc new_disc) {
        if(disc_count < capacity) {
            new_disc.status = disc_status::on_tower;
            new_disc.tower_id = id;

            discs[disc_count] = new_disc;
            disc_count++;
        }
    }

    disc peek() {
        if(disc_count > 0) {
            return discs[disc_count - 1];
        } else {
            throw std::runtime_error("Tower " + std::to_string(id) + " is empty");
        }
    }

    void pop() {
        if(disc_count > 0) {
            discs[disc_count - 1].status = disc_status::invalid;
            discs[disc_count - 1].tower_id = -1;
            disc_count--;
        }
    }

    std::pair<int, const disc*> get_discs() {
        return {disc_count, discs};
    }

    tower& operator=(const tower& other) {
        if(this == &other) return *this;
        
        delete[] discs;

        id = other.id;
        disc_count = other.disc_count;
        capacity = other.capacity;

        if(capacity > 0) {
            discs = new disc[capacity];
            for(int i = 0; i < capacity; i++) {
                discs[i] = other.discs[i];
            }
        } else {
            discs = nullptr;
        }
        
        return *this;
    }

    disc* begin() {
        return discs;
    }
    disc* end() {
        return discs + (sizeof(disc) * capacity);
    }

    bool empty() {
        return (!disc_count);
    }

    ~tower() {
        delete[] discs;
    }
};



void print_towers(std::pair<disc, std::vector<tower>> game_context, int pointer_pos);
op get_op(const char& in, std::pair<disc, std::vector<tower>> game_context, int pointer_pos);

void game_loop(int disc_count);


int main() {

    game_loop(5);

    return 0;
}



void game_loop(int disc_count) {
    if(disc_count > 5 || disc_count < 2) {
        printf("Disc Count can only be 5 - 2");
        disc_count = 5;
    }

    /* GAME LOOP CONTEXT */
    bool running = true;
    // the towers get_op(argv) will read;
    std::vector<tower> towers;
    for(int i = 0; i < TOWER_COUNT; i++) {
        towers.emplace_back(i, disc_count);
    }
    printf("Towers created\n");

    // create an invalid disc
    // put the popped discs here
    disc on_hand = disc();
    printf("Invisible Disc created\n");

    // pointer to which you want to make a move
    // pointer_pos: 0 or 1 or 2
    int pointer_pos = 0;

    // counts the moves
    int moves = 0;
    // same context as the move
    int prev_pointer_pos = pointer_pos;


    while(running) {
        std::pair<disc, std::vector<tower>> game_context = std::make_pair(on_hand, towers);
        /* uodate the tower (from last iter)*/
        print_towers(game_context, pointer_pos);

        /* get input */
        char in = getch();
        op make = get_op(in, game_context, pointer_pos);

        if(make == op::stop) {
            break;
        } else if(make == op::move_left) {
            pointer_pos--;
        } else if (make == op::move_right) {
            pointer_pos++;
        } else if (make == op::place_disc) {
            towers[pointer_pos].push(on_hand);
            on_hand = disc(); // reset hand to invalid

            /* MOVE INCREMENT LOGIC*/
            // only update the move if you moved the disc on a different tower
            moves += (prev_pointer_pos != pointer_pos);
            /* MOVE INCREMENT LOGIC*/
            
        } else if(make == op::take_disc) {
            if(!towers[pointer_pos].empty()) {
                on_hand = towers[pointer_pos].peek();
                on_hand.status = disc_status::hand;
                towers[pointer_pos].pop();

                /* MOVE INCREMENT LOGIC*/
                prev_pointer_pos = pointer_pos;
                /* MOVE INCREMENT LOGIC*/
            }
        }

        /* check if all the disks are on towers[2] (the last tower) */
        if(towers[TOWER_COUNT - 1].disc_count == disc_count) {
            printf(" >--- SCORE: %d ---< \n", moves);
            system("pause");

            running = false;
        }

        system("cls");
    }
}


op get_op(const char& in , std::pair<disc, std::vector<tower>> game_context, int pointer_pos) {
    op move = op::invalid_input;

    switch (in) {
        case 'a':
            if (pointer_pos > 0)
                move = op::move_left;
            break;

        case 'd':
            if (pointer_pos < TOWER_COUNT - 1)
                move = op::move_right;
            break;

        case ' ': // space: take or place
            if (game_context.first.status == disc_status::hand) {
                // place: allowed if target empty or top weight > hand weight
                if (game_context.second[pointer_pos].disc_count == 0) {
                    move = op::place_disc;
                } else {
                    try {
                        if (game_context.second[pointer_pos].peek().weight > game_context.first.weight)
                            move = op::place_disc;
                    } catch (...) {
                        // peek threw, but treat as place allowed
                        move = op::place_disc;
                    }
                }
            } else {
                // take: allowed if tower not empty
                if (game_context.second[pointer_pos].disc_count > 0)
                    move = op::take_disc;
            }
            break;

        case 'p':
            move = op::stop;
            break;

        default:
            break;
    }

    return move;
}



void print_towers(std::pair<disc, std::vector<tower>> game_context, int pointer_pos) {

    // disc on hand print
    if(game_context.first.status == disc_status::hand) {
        for(int i = 0; i < TOWER_COUNT; i++) {
            if(i == pointer_pos) {
                printf("%s", disc_pads[game_context.first.weight]);
                printf("%s", disc_visual[game_context.first.weight]);
                printf("%s", disc_pads[game_context.first.weight]);
            } else {
                printf("%s", disc_blank_padding);
            }
        }
    } else {
        printf("%s%s%s", disc_blank_padding, disc_blank_padding, disc_blank_padding);
    }
    printf("\n");

    // towers print
    printf("                                                   \n");
    for(int i = 0; i < TOWER_COUNT; i++) {
        if(pointer_pos == i) printf("%s", "\033[91m");
        printf("        |        ");
        printf("%s", "\033[0m");
    }
    printf("\n");

    // discs in the tower print
    for(int i = game_context.second[0].capacity - 1; i >= 0; i--) {
        for(int j = 0; j < TOWER_COUNT; j++) {
            if(game_context.second[j].discs[i].weight != -1) {
                printf("%s", disc_pads[game_context.second[j].discs[i].weight]);
                if(pointer_pos == j) printf("%s", "\033[91m");
                printf("%s", disc_visual[game_context.second[j].discs[i].weight]);
                printf("%s", "\033[0m");
                printf("%s", disc_pads[game_context.second[j].discs[i].weight]);
            } else {
                printf("%s", empty_pads);
                if(pointer_pos == j) printf("%s", "\033[91m");
                printf("%s", empty);
                printf("%s", "\033[0m");
                printf("%s", empty_pads);
            }
        }
        printf("\n");
    }

    // footing print
    for(int i = 0; i < TOWER_COUNT; i++) {
        printf("========");
        if(pointer_pos == i) printf("%s", "\033[91m");
        printf("|");
        printf("%s", "\033[0m");
        printf("========");
    }
    printf("\n");


    // debug print
    /* 
    for(const auto& tower : game_context.second) {
        //     "        |        "
        printf("      {[%d]}      ", tower.disc_count);
    }
    if(game_context.first.status == disc_status::hand) {
        printf("\nHAND\n");
    } else if(game_context.first.status == disc_status::invalid) {
        printf("\nINVALID\n");
    } else if(game_context.first.status == disc_status::on_tower) {
        printf("\nON TOWER\n");
    }
    */

    /*
    printf("                                                   "); // disc@i?,     disc@i?,     disc@i?
    printf("                                                   "); //             just print
    printf("        |                |                |        "); //             just print
    printf("       ( )              ( )              ( )       "); // tower[0][0], tower[1][0], tower[2][0]
    printf("      (   )            (   )            (   )      "); // tower[0][1], tower[1][1], tower[2][1]
    printf("     (     )          (     )          (     )     "); // tower[0][2], tower[1][2], tower[2][2]
    printf("    (       )        (       )        (       )    "); // tower[0][3], tower[1][3], tower[2][3]
    printf("   (         )      (         )      (         )   "); // tower[0][4], tower[1][4], tower[2][4]
    printf("========|================|================|========"); //             just print
    // ^^ 51 characters long ^^
    // discs are 11 - 3 characters long
    */
}



