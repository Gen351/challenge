#include<iostream>
#include<conio.h>


int main() {

    while(true) {
        if(_kbhit()) {
            char in = getch();

            if(in == 27) {
                break;
            }

            printf("%c ", in);
        }
    }


}
