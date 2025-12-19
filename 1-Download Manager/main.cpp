#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>


//
//
//       N O T   B Y   M E   ! ! ! ! ! ! !
//
//





#pragma comment(lib, "ws2_32.lib")

int main() {
    // 1. SETUP
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    std::string hostname = "speedtest.tele2.net";
    std::string filepath = "/1MB.zip";
    std::string output_file = "test_download.zip";

    struct addrinfo hints = {}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    getaddrinfo(hostname.c_str(), "80", &hints, &res);

    SOCKET sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    connect(sock, res->ai_addr, (int)res->ai_addrlen);

    // 2. REQUEST
    std::string request = "GET " + filepath + " HTTP/1.1\r\n" +
                          "Host: " + hostname + "\r\n" +
                          "Connection: close\r\n\r\n";
    send(sock, request.c_str(), request.length(), 0);

    // 3. RECEIVE & PARSE
    std::ofstream outfile(output_file, std::ios::binary);
    char buffer[4096];
    int bytesReceived;
    int totalBytes = 0;
    
    // This flag tracks if we found the end of the header yet
    bool headerFinished = false; 
    
    // We might need to store partial data if the header is split across two packets
    std::string responseBuffer = ""; 

    std::cout << "Downloading..." << std::endl;

    while ((bytesReceived = recv(sock, buffer, 4096, 0)) > 0) {
        if (!headerFinished) {
            // Append current chunk to our temporary buffer string
            responseBuffer.append(buffer, bytesReceived);

            // Look for the "Double New Line" (\r\n\r\n)
            size_t headerEnd = responseBuffer.find("\r\n\r\n");

            if (headerEnd != std::string::npos) {
                // FOUND IT! The file data starts 4 bytes after the match
                headerFinished = true;
                
                // Calculate where the real data starts
                size_t bodyStart = headerEnd + 4;
                
                // Write the rest of this chunk to the file
                // (responseBuffer.size() - bodyStart) is the amount of data left
                outfile.write(responseBuffer.c_str() + bodyStart, responseBuffer.size() - bodyStart);
                
                totalBytes += (responseBuffer.size() - bodyStart);
                std::cout << "\rHeader skipped! Saving data..." << std::flush;
            }
        } else {
            // Header is already gone, just write raw data directly to file
            outfile.write(buffer, bytesReceived);
            totalBytes += bytesReceived;
            std::cout << "\rDownloaded: " << totalBytes << " bytes" << std::flush;
        }
    }

    std::cout << "\nDownload Complete! Saved " << totalBytes << " bytes.\n";

    outfile.close();
    closesocket(sock);
    freeaddrinfo(res);
    WSACleanup();

    return 0;
}