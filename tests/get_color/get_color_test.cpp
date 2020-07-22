//
// Created by Komorowicz David on 2020. 07. 22..
//
#include "fmt/core.h"

#include <bundleadjust/KinectDataloader.h>

int main(){
    KinectDataloader* data = new KinectDataloader("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");
    auto color = data->getPointColor(0);

    data->visualizeMatch(0, 1);

    fmt::print("color {} {} {}", color(0), color(1), color(2));
}
