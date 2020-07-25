//
// Created by Komorowicz David on 2020. 07. 22..
//
#include "fmt/core.h"

#include <bundleadjust/KinectDataloader.h>

int main(){
    KinectDataloader* data = new KinectDataloader("/Users/dawars/Documents/university/master/TUM/1st_semester/3d_scanning/group_project/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");

    Eigen::Matrix4f mat;
    mat.setRandom();

    data->setEstimatedPose(1, mat);
    auto mat2 = Eigen::Map<const Eigen::Matrix4f>(data->getEstimatedPose(1));

    return mat2.isApprox(mat, 1e-6) != true;
}
