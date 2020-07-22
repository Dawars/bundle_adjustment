//
// Created by Komorowicz David on 2020. 07. 22..
//
#include "fmt/core.h"
#include <iostream>

#include <bundleadjust/KinectDataloader.h>

int main(){
    KinectDataloader* data = new KinectDataloader("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");

    auto numFrames = data->getNumFrames();
    Eigen::MatrixXi connectivity(numFrames, numFrames);


    for (int i = 0; i < numFrames; ++i) {
        for (int j = 0; j < numFrames; ++j) {
            connectivity(i,j) = 0;
        }
    }

    for (int i = 0; i < numFrames; ++i) {
        for (int j = 0; j < i; ++j) {

            auto obs1 = data->correspondenceFinder->getCamObs(i);
            auto obs2 = data->correspondenceFinder->getCamObs(j);

            for(auto &pt1 : obs1){
                for (auto &pt2 : obs2){
                    if(data->getObsPoint(pt1) != -1 && data->getObsPoint(pt1) == data->getObsPoint(pt2)) {
                        connectivity(i, j) += 1;
                        connectivity(j, i) += 1;
                    }
                }
            }
        }
    }

    std::cout << connectivity << std::endl;
}
