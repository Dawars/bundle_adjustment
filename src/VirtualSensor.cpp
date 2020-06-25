// Taken and modified from exercise 1
#include "VirtualSensor.h"

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats

VirtualSensor::VirtualSensor() : m_currentIdx(-1), m_increment(10) {

}

VirtualSensor::~VirtualSensor() {
}

bool VirtualSensor::Init(const std::string &datasetDir) {
    m_baseDir = datasetDir;

    // read filename lists
    if (!ReadFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps)) return false;
    if (!ReadFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps)) return false;

    // read tracking
    if (!ReadTrajectoryFile(datasetDir + "groundtruth.txt", m_trajectory, m_trajectoryTimeStamps)) return false;

    if (m_filenameDepthImages.size() != m_filenameColorImages.size()) return false;

    // image resolutions
    m_colorImageWidth = 640;
    m_colorImageHeight = 480;
    m_depthImageWidth = 640;
    m_depthImageHeight = 480;

    // intrinsics
    m_colorIntrinsics << 525.0f, 0.0f, 319.5f,
            0.0f, 525.0f, 239.5f,
            0.0f, 0.0f, 1.0f;

    m_depthIntrinsics = m_colorIntrinsics;

    m_colorExtrinsics.setIdentity();
    m_depthExtrinsics.setIdentity();

    m_currentIdx = -1;
    return true;
}

bool VirtualSensor::ProcessNextFrame() {
    if (m_currentIdx == -1) m_currentIdx = 0;
    else m_currentIdx += m_increment;

    if ((unsigned int) m_currentIdx >= (unsigned int) m_filenameColorImages.size()) return false;

    std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]" << std::endl;

    //opencv mat
    rgbImage = cv::imread(m_baseDir + m_filenameColorImages[m_currentIdx]);
    depthImage = cv::imread(m_baseDir + m_filenameDepthImages[m_currentIdx],
                            cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    depthImage.convertTo(depthImage, CV_32FC1); // or CV_32F works (too)
//    std::cout << depthImage << std::endl;

    // depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
    depthImage = depthImage / 5000.;

    // if image is 0, set to -inf, otherwise /5000
    float oldValue = 0;
    float newValue = -std::numeric_limits<float>::infinity();

    depthImage.setTo(newValue, depthImage == oldValue);

    // find transformation (simple nearest neighbor, linear search)
    double timestamp = m_depthImagesTimeStamps[m_currentIdx];
    double min = std::numeric_limits<double>::max();
    int idx = 0;
    for (unsigned int i = 0; i < m_trajectory.size(); ++i) {
        double d = fabs(m_trajectoryTimeStamps[i] - timestamp);
        if (min > d) {
            min = d;
            idx = i;
        }
    }
    m_currentTrajectory = m_trajectory[idx];


    return true;
}

unsigned int VirtualSensor::GetCurrentFrameCnt() {
    return (unsigned int) m_currentIdx;
}

// get current color data
cv::Mat VirtualSensor::GetColorRGBX() const {
    return rgbImage;
}

// get current depth data
cv::Mat VirtualSensor::GetDepth() const {
    return depthImage;
}

// color camera info
Eigen::Matrix3f VirtualSensor::GetColorIntrinsics() {
    return m_colorIntrinsics;
}

Eigen::Matrix4f VirtualSensor::GetColorExtrinsics() {
    return m_colorExtrinsics;
}

unsigned int VirtualSensor::GetColorImageWidth() {
    return m_colorImageWidth;
}

unsigned int VirtualSensor::GetColorImageHeight() {
    return m_colorImageHeight;
}

// depth (ir) camera info
Eigen::Matrix3f VirtualSensor::GetDepthIntrinsics() {
    return m_depthIntrinsics;
}

Eigen::Matrix4f VirtualSensor::GetDepthExtrinsics() {
    return m_depthExtrinsics;
}

unsigned int VirtualSensor::GetDepthImageWidth() {
    return m_colorImageWidth;
}

unsigned int VirtualSensor::GetDepthImageHeight() {
    return m_colorImageHeight;
}

// get current trajectory transformation
Eigen::Matrix4f VirtualSensor::GetTrajectory() {
    return m_currentTrajectory;
}

bool VirtualSensor::ReadFileList(const std::string &filename, std::vector<std::string> &result,
                                 std::vector<double> &timestamps) {
    std::ifstream fileDepthList(filename, std::ios::in);
    if (!fileDepthList.is_open()) return false;
    result.clear();
    timestamps.clear();
    std::string dump;
    std::getline(fileDepthList, dump);
    std::getline(fileDepthList, dump);
    std::getline(fileDepthList, dump);
    while (fileDepthList.good()) {
        double timestamp;
        fileDepthList >> timestamp;
        std::string filename;
        fileDepthList >> filename;
        if (filename == "") break;
        timestamps.push_back(timestamp);
        result.push_back(filename);
    }
    fileDepthList.close();
    return true;
}

bool VirtualSensor::ReadTrajectoryFile(const std::string &filename, std::vector<Eigen::Matrix4f> &result,
                                       std::vector<double> &timestamps) {
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open()) return false;
    result.clear();
    std::string dump;
    std::getline(file, dump);
    std::getline(file, dump);
    std::getline(file, dump);

    while (file.good()) {
        double timestamp;
        file >> timestamp;
        Eigen::Vector3f translation;
        file >> translation.x() >> translation.y() >> translation.z();
        Eigen::Quaternionf rot;
        file >> rot;

        Eigen::Matrix4f transf;
        transf.setIdentity();
        transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
        transf.block<3, 1>(0, 3) = translation;

        if (rot.norm() == 0) break;

        transf = transf.inverse().eval();

        timestamps.push_back(timestamp);
        result.push_back(transf);
    }
    file.close();
    return true;
}