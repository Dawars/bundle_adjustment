//
// Created by Komorowicz David on 2020. 06. 15..
//
#include <iostream>

#include "harris/harris.h"


int main(){
    Harris h(nullptr, 10, 10);
    std::cout << h.getNumCores() << std::endl;
    return 0;
}