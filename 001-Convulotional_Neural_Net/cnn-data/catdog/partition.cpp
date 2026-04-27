#include<fstream>
#include <cctype>
#include<iostream>
#include <vector>
#include<time.h>
#include<random>
#include<algorithm>
#include<string>

#include<sstream>

const std::string joinedFile = "./catdog_breeds.csv";
const std::string trainFile = "breeds_shuffled_train.csv";
const std::string testFile = "breeds_shuffled_test.csv";

std::string getFirstElem(const std::string& in, const char delim);
void shuffle(std::vector<std::string>& data);

int main() {
    srand(time(0));

    std::ifstream readFile(joinedFile);

    std::vector<std::string> datas;

    std::string temp;
    if(readFile.is_open()) {
        std::cout << "Reading Data\n";
        int i = 0;
        while(std::getline(readFile, temp)) {
            // remove the label
            if(i++ == 0) continue;
            if(temp.empty()) continue;

            datas.emplace_back(temp);
        }
        readFile.close();

        std::cout << "Sort Data Based Off of Label\n";
        // sort based off of the label
        std::sort(datas.begin(), datas.end(), [](const std::string& a, const std::string& b) {
            return std::stoi(getFirstElem(a, ',')) < std::stoi(getFirstElem(b, ','));
        });

        std::cout << "Partition Data\n";
        // partition
        std::vector<std::string> train;
        std::vector<std::string> test;

        int currClass = std::stoi(getFirstElem(datas[0], ','));
        int counter = 0;
        for(int i = 0; i < datas.size(); i++) {
            int tempClass = std::stoi(getFirstElem(datas[i], ','));
            if(tempClass != currClass) {
                counter = 0;
                currClass = tempClass;
            }

            if(counter < 140) {
                train.emplace_back(datas[i]);
            } else {
                test.emplace_back(datas[i]);
            }
            counter++;
        }

        std::cout << "Shuffling\n";
        // shuffle train and test
        shuffle(train);
        shuffle(test);

        std::cout << "Write Train\n";
        // write train
        std::ofstream writeFile(trainFile);
        if(writeFile.is_open()) {
            // iterate over the classes and write on file
            int i = 0;
            for (const auto& trainData : train) {
                writeFile << trainData << "\n";
            }
            writeFile.close();
        } else {
            std::cerr << "Cannot open: " << trainFile << std::endl;
        }

        std::cout << "Write Test\n";
        // write test
        writeFile.open(testFile);
        if(writeFile.is_open()) {
            // iterate over the classes and write on file
            for (const auto& testData : test) {
                writeFile << testData << "\n";
            }
            writeFile.close();
        } else {
            std::cerr << "Cannot open: " << testFile << std::endl;
        }


    } else {
        std::cerr << "Cannot open: " << joinedFile << std::endl;
    }

    return 0;
}



std::string getFirstElem(const std::string& in, const char delim) {
    std::string firstElem;
    
    std::istringstream inStream(in);
    std::getline(inStream, firstElem, delim);

    return firstElem;
}

void shuffle(std::vector<std::string>& data) {
    for(int i = data.size() - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        std::swap(data[i], data[j]);
    }
}