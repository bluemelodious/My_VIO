#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc {
public:
    TicToc() {
        begin = std::chrono::system_clock::now();
    }
    double toc() {
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> cost_time = end - begin;
        return cost_time.count()*1000;
    }   
private:
    std::chrono::time_point<std::chrono::system_clock> begin;

};