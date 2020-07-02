//
// Created by Komorowicz David on 2020. 06. 15..
//
#include <iostream>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BalDataloader.h"
#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/PointMatching.h"


int main() {

    OnlinePointMatcher pm;
    pm.read_images("/home/andrew/repos/tum/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/rgb");
    pm.match_with_frame(4);


    return 0;
}