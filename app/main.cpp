//
// Created by Komorowicz David on 2020. 06. 15..
//
#include <iostream>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BalDataloader.h"


int main(){
    BalDataloader data("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/bal/ladybug/problem-49-7776-pre.txt");

    BundleAdjustment ba{data};

    ba.createProblem();
    ba.solve();

    return 0;
}