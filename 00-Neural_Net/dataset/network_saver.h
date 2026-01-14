#pragma once
#include<iostream>

#include <sstream>   // <-- for ostringstream
#include <iomanip>   // <-- for setprecision

#include <vector>
#include <string>

#include<fstream>


namespace NetworkSaver {

    inline void saveNetwork(const std::vector<std::string>& formattedNetwork, const std::string& networkName) {
        int iter = 0;
        std::string filename;

        while (true) {
            filename = networkName + std::to_string(iter) + ".nn";

            std::ifstream test(filename);
            if (!test.good()) {
                break; // file does NOT exist → good to use
            }

            iter++;
        }

        std::ofstream write(filename);
        if (!write) return; // or throw

        for (size_t i = 0; i < formattedNetwork.size(); i++) {
            write << formattedNetwork[i];
            if (i + 1 != formattedNetwork.size())
                write << '\n';
        }

        std::cout << "\nNN saved: " << filename << '\n';
    }

};