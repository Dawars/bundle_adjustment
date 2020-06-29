//
// Created by Komorowicz David on 2020. 06. 15..
//
#include <iostream>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BalDataloader.h"
#include "bundleadjust/KinectDataloader.h"


int main(){
//    BalDataloader data("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/bal/venice/problem-52-64053-pre.txt");
//
//    BundleAdjustment ba{data};
//
//    ba.createProblem();
//    ba.solve();
//
//    ba.writeMesh("veniceGroundTruth.off");
//    ba.writeCamerasMesh("veniceCameraGroundTruth.off");


    ba.createProblem();
    ba.solve();

    KinectDataloader kinectDataloader("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");

    return 0;
}