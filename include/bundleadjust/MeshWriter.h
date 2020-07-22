//
// Created by Komorowicz David on 2020. 07. 22..
//

#pragma once

#include <string>

#include <Eigen/Dense>

class MeshWriter {
public:
    static void WriteToPLYFile(std::string filename, std::vector<Eigen::Vector3f> points,  std::vector<Eigen::Vector3i> colors, std::vector<Eigen::Vector3f> cams={});
};

