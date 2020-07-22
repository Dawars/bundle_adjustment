

#include "bundleadjust/ProcrustesAligner.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

int main() {

    std::vector<Eigen::Vector3f> source_points;
    Eigen::Vector3f a_source;
    Eigen::Vector3f b_source;
    Eigen::Vector3f c_source;
    Eigen::Vector3f d_source;
    a_source << 0, 0, 2;
    b_source << -1, 0, 3;
    c_source << -1, 2, 1;
    d_source << 2, 3, 3;
    source_points.push_back(a_source);
    source_points.push_back(b_source);
    source_points.push_back(c_source);
    source_points.push_back(d_source);

    std::vector<Eigen::Vector3f> target_points;
    Eigen::Vector3f a_target;
    Eigen::Vector3f b_target;
    Eigen::Vector3f c_target;
    Eigen::Vector3f d_target;
    a_target << 2, 0, 3;
    b_target << 3, 0, 4;
    c_target << 1, 2, 4;
    d_target << 3, 3, 1;
    target_points.push_back(a_target);
    target_points.push_back(b_target);
    target_points.push_back(c_target);
    target_points.push_back(d_target);
    


    ProcrustesAligner aligner;

    Eigen::Matrix4f estimated_pose = aligner.estimatePose(target_points, source_points);

    Eigen::Matrix3f rotation_matrix;
    rotation_matrix << estimated_pose.block(0, 0, 3, 3);
    Eigen::AngleAxis<float> r = Eigen::AngleAxis<float>(rotation_matrix);

    return -1;
}