#include<iostream>
#include<vector>
#include<stdexcept>
#include<conio.h>

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
        tower_id = 0;
        status = disc_status::invalid;
        weight = -1;
    }
    disc(disc_status stat, int w) {
        status = stat;
        weight = w;
    }
    disc(int w) {
        status = disc_status::invalid;
        weight = w;
    }
};

struct tower {
    int id;
    int disc_count = 0;
    disc* discs;
    
    tower(int i, int count) {
        id = i;

        if(i == 0) {
            disc_count = count;
            discs = new disc[disc_count];
            for(int w = 0; w < disc_count; w++) {
                discs[disc_count - w] = disc(w);
            }
        }    
    }

    void push(disc new_disc, int max_disc) {
        if(disc_count < max_disc) {
            discs[disc_count++] = new_disc;
        }
    }

    disc peek() {
        if(disc_count > 0) {
            return discs[disc_count];
        } else {
            throw std::runtime_error("Tower " + std::to_string(id) + " is emprty");
        }
    }

    void pop() {
        if(disc_count > 0) {
            discs[disc_count--].status = disc_status::invalid;
        }
    }

    std::pair<int, disc*> get_discs() {
        return {disc_count, discs};
    }

    bool empty() {
        return (!disc_count);
    }

    ~tower() {
        delete[] discs;
    }
};





void print_towers(std::pair<disc, std::vector<tower>> game_context, int pointer_pos);
op get_op(std::pair<disc, std::vector<tower>> game_context, int pointer_pos);

void game_loop();


int main() {

    std::string test = "(       )";

    std::cout << test.length() << "\n";

    return 0;
}



void game_loop(int disc_count) {
    
    /* GAME LOOP CONTEXT */
    bool running = true;
    // the towers get_op(argv) will read;
    std::vector<tower> towers = {tower(0, disc_count), tower(1, disc_count), tower(2, disc_count)};
    
    // create an invalid disc
    // put the popped discs here
    disc on_hand = disc();

    // pointer to which you want to make a move
    // pointer_pos: 0 or 1 or 2
    int pointer_pos = 0;


    while(running) {

        if(_kbhit()) {
            std::pair<disc, std::vector<tower>> game_context = std::make_pair(on_hand, towers);
            op make = get_op(game_context, pointer_pos);
            
            if(make == op::move_left) {
                if(pointer_pos > 1) {
                    pointer_pos--;
                }
            } else if (make == op::move_right) {
                if(pointer_pos < 2) {
                    pointer_pos++;
                }
            } else if (make == op::place_disc) {
                towers[pointer_pos].push(on_hand, disc_count);
            } else if(make == op::take_disc) {
                if(!towers.empty()) {
                    on_hand = towers[pointer_pos].peek();
                    towers[pointer_pos].pop();
                }
            }

            /* uodate the tower */
            // print_towers()
        }

        /* check if all the disks are on towers[2] */

        running = true;
    }

}


op get_op(std::pair<disc, std::vector<tower>> game_context, int pointer_pos) {
    // read the buffer from kbhit
    char in = _getch();
    // remove unnecessary chars
    // only read in[0]
    op move = op::invalid_input;

    switch(in) {
        case 'a':
            move = op::move_left;
            if(game_context.first.status != disc_status::hand)
                move = op::invalid_input;
            else if(game_context.first.status == disc_status::invalid)
                move = op::invalid_input;
            else if(pointer_pos < 1) 
                move = op::invalid_input;
            break;

        case 'd':
            move = op::move_right;
            if(game_context.first.status != disc_status::hand)
                move = op::invalid_input;
            else if(game_context.first.status == disc_status::invalid)
                move = op::invalid_input;
            else if(pointer_pos > 1)
                move - op::invalid_input;
            break;

        case ' ':
            move = op::place_disc;
            if(game_context.first.status == disc_status::on_tower)
                move = op::take_disc;
            else if(game_context.first.status != disc_status::hand)
                move = op::invalid_input;
            // can't put bigger discs on smaller ones
            else if(game_context.first.status != disc_status::hand && game_context.second[pointer_pos].peek().weight < game_context.first.weight)
                move = op::invalid_input;
            break;

        case 'p':
            move = op::stop;
            break;
        
        default:
            move == op::invalid_input;
            break;
    }

    return move;
}



void print_towers(std::pair<disc, std::vector<tower>> game_context, int pointer_pos) {

    

    
    printf("                                                   ");
    printf("                                                   ");
    printf("        |                |                |        ");
    printf("       ( )              ( )              ( )       ");
    printf("      (   )            (   )            (   )      ");
    printf("     (     )          (     )          (     )     ");
    printf("    (       )        (       )        (       )    ");
    printf("   (         )      (         )      (         )   ");
    printf("========|================|================|========");
    // ^^ 51 characters long ^^

    // discs are 11 - 3 characters long
}