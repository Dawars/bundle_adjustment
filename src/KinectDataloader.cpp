//
// Created by Komorowicz David on 2020. 06. 25..
//

#include <cmath>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <ceres/rotation.h>
#include <fmt/core.h>
#include <bundleadjust/MeshWriter.h>

#include "VirtualSensor.h"
#include "bundleadjust/PointMatching.h"
#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/HarrisDetector.h"
#include "bundleadjust/ShiTomasiDetector.h"
#include "bundleadjust/SiftDetector.h"
#include "bundleadjust/ProcrustesAligner.h"

using namespace cv::xfeatures2d;

void visualize(cv::Mat image, std::vector<cv::KeyPoint> featurePoints) {

    std::string name{"Detected corners"};
    cv::Mat out(image.size(), image.type());
    cv::drawKeypoints(image, featurePoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::namedWindow(name);
    cv::imshow(name, out);
    cv::waitKey(0);
}

void KinectDataloader::visualizePointMatch(const int frame_one, const int frame_two, const int point_index, const int match_index) const {
    std::vector<cv::DMatch> matches;
    cv::Mat img1 = this->colorImages[frame_one];
    cv::Mat img2 = this->colorImages[frame_two];

    auto matcher = this->correspondenceFinder->matcher;

    auto kpts = this->correspondenceFinder->getKeyPoints();

    auto kpts1 = kpts[frame_one];
    auto kpts2 = kpts[frame_two];

    for (int i = 0; i < kpts1.size(); ++i) {
        int obsIndex1 = correspondenceFinder->getObsIndex(frame_one, i);

        if (this->correspondenceFinder->getObsPoint(obsIndex1) != -1) {
            for (int j = 0; j < kpts2.size(); ++j) {
                int obsIndex2 = correspondenceFinder->getObsIndex(frame_two, j);

                if (this->correspondenceFinder->getObsPoint(obsIndex1) ==
                    this->correspondenceFinder->getObsPoint(obsIndex2)) {
                    matches.push_back(cv::DMatch(i, j, 1.));
                }
            }
        }
    }

    std::vector<cv::DMatch> match;
    match.push_back(matches[match_index]);
    cv::Mat out;
    cv::drawMatches(img1, kpts1, img2, kpts2, match, out);

    std::string name(fmt::format("Matches between frames {} & {}", frame_one, frame_two));
    cv::namedWindow(name);
    cv::imshow(name, out);
    cv::waitKey(0);
}

void KinectDataloader::visualizeMatch(const int frame_one, const int frame_two) const {
    std::vector<cv::DMatch> matches;
    cv::Mat img1 = this->colorImages[frame_one];
    cv::Mat img2 = this->colorImages[frame_two];

    auto matcher = this->correspondenceFinder->matcher;

    auto kpts = this->correspondenceFinder->getKeyPoints();

    auto kpts1 = kpts[frame_one];
    auto kpts2 = kpts[frame_two];

    for (int i = 0; i < kpts1.size(); ++i) {
        int obsIndex1 = correspondenceFinder->getObsIndex(frame_one, i);

        if (this->correspondenceFinder->getObsPoint(obsIndex1) != -1) {
            for (int j = 0; j < kpts2.size(); ++j) {
                int obsIndex2 = correspondenceFinder->getObsIndex(frame_two, j);

                if (this->correspondenceFinder->getObsPoint(obsIndex1) ==
                    this->correspondenceFinder->getObsPoint(obsIndex2)) {
                    matches.push_back(cv::DMatch(i, j, 1.));
                }
            }
        }
    }

    cv::Mat out;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, out);

    std::string name(fmt::format("Matches between frames {} & {}", frame_one, frame_two));
    cv::namedWindow(name);
    cv::imshow(name, out);
    cv::waitKey(0);
}


KinectDataloader::KinectDataloader(const std::string &datasetDir, bool initGroundTruth)
        : initGroundTruth(initGroundTruth) {
//    std::unordered_map<std::string, float> params = {
//            {"blockSize",    2.0},
//            {"apertureSize", 3.0},
//            {"k",            0.04},
//            {"thresh",       200}
//    };
//
//    SiftDetector detector;
//    HarrisDetector detector;
//    ShiTomasiDetector detector;

    auto detector = cv::SIFT::create(); // TODO: Extend and use FeatureDetector wrapper
    auto extractor = cv::SIFT::create();
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    correspondenceFinder = new OnlinePointMatcher{detector, extractor, matcher, {{"ratioThreshold", 0.8},
                                                                                 {"ransacEps", 1e1}}};
    this->sensor = new VirtualSensor{};
    if(!sensor->Init(datasetDir)) { throw std::invalid_argument("Kinect dataset could not be loaded");}

    this->intrinsics = sensor->GetColorIntrinsics();

//    for (int i = 0; i < 3 && sensor->ProcessNextFrame(); ++i) {
    while (sensor->ProcessNextFrame()) {
        auto color = sensor->GetColor();
        auto depth = sensor->GetDepth();

        // Future improvement https://vision.unipv.it/corsi/computervision/secondchoice/0279%20Depth%20Image%20Enhancement%20for%20Kinect%20Using%20Region%20Growing%20and%20Bilateral%20-%20Copia.pdf
        cv::Mat depthFiltered;
        cv::bilateralFilter(depth, depthFiltered, 5, 3, 1.2);

//        cv::imshow("depth", depth);
//        cv::imshow("depthFiltered", depthFiltered);
//        cv::waitKey(0);

        colorImages.push_back(color);
        depthImages.push_back(depthFiltered);

        correspondenceFinder->extractKeypoints(color);
    }


    correspondenceFinder->matchKeypoints();

    setupPointDepth();

    this->estimatedPoses = new float[getNumFrames()*16];
}

KinectDataloader::~KinectDataloader() {
    delete this->correspondenceFinder;
    delete this->sensor;
    delete this->estimatedPoses;
}

int KinectDataloader::getObsCam(int index) const {
    return this->correspondenceFinder->getObsCam(index);
}

int KinectDataloader::getObsPoint(int index) const {
    return this->correspondenceFinder->getObsPoint(index);
}

int KinectDataloader::getNumPoints() const {
    return this->correspondenceFinder->getNumPoints();
}

std::vector<cv::Point2f> KinectDataloader::getObservations() const {
    return this->correspondenceFinder->getObservations();
}

int KinectDataloader::getNumObservations() const {
    return this->correspondenceFinder->getNumObservations();
}

int KinectDataloader::getNumFrames() const {
    return this->correspondenceFinder->getNumFrames();
}

bool KinectDataloader::isColorAvailable() const {
    return true;
}

bool KinectDataloader::isDepthAvailable() const {
    return true;
}

cv::Mat KinectDataloader::getColor(int frameId) const {
    return this->colorImages[frameId];
}

cv::Mat KinectDataloader::getDepth(int frameId) const {
    return this->depthImages[frameId];
}

void KinectDataloader::setupPointDepth() {
    Eigen::Matrix3f instrinsicsInv = this->intrinsics.inverse();

    for (int i = 0; i < getNumFrames(); ++i) {
        std::vector<Eigen::Vector3f> pointsToPrint;
        std::vector<Eigen::Vector3i> colorsToPrint;

        auto &kps = this->correspondenceFinder->keypoints[i];
        auto num_current_points = kps.size();

        const int frame_width = depthImages[i].size[1];
        const int frame_height = depthImages[i].size[0];

        // build x, y, z observations
        for (int j = 0; j < num_current_points; j++) {
            double x_obs = kps[j].pt.x;
            double y_obs = kps[j].pt.y;

            Eigen::Vector3f image_point;
            image_point << x_obs, y_obs, 1;

            assert(x_obs <= frame_width);
            assert(y_obs <= frame_height);

            float depth = depthImages[i].at<float>(y_obs, x_obs);
            Eigen::Vector3f cameraLine = instrinsicsInv * image_point;
            Eigen::Vector4f cameraPoint; // Andrew: don't think homogenous will be necessary
            cameraPoint << depth * cameraLine, 1;

            //std::cout << cameraPoint << "\n\n";

            // needs to be in camera space so that units match (x,y) & z
            x.push_back(cameraPoint(0));
            y.push_back(cameraPoint(1));
            z.push_back(cameraPoint(2));

            if (!std::isinf(depth)) {
                Eigen::Vector3f point_vector;
                point_vector << cameraPoint(0), cameraPoint(1), cameraPoint(2);
                pointsToPrint.push_back(point_vector);

                Eigen::Vector3i rgb_vector;
                cv::Vec3b color = colorImages[i].at<cv::Vec3b>(y_obs, x_obs);
                rgb_vector << color[2], color[1], color[0];
                colorsToPrint.push_back(rgb_vector);
            }
        }

        // saving mesh
        //MeshWriter::WriteToPLYFile(fmt::format("frame{}.ply", i), pointsToPrint, colorsToPrint);

    }
}

void KinectDataloader::initialize(double *R, double *T, double *intrinsics, double *X) {
    for (int i = 0; i < this->getNumFrames(); ++i) {
        intrinsics[6 * i + 0] = this->intrinsics(0, 0);
        intrinsics[6 * i + 1] = this->intrinsics(1, 1);
        intrinsics[6 * i + 2] = this->intrinsics(0, 2);
        intrinsics[6 * i + 3] = this->intrinsics(1, 2);
        intrinsics[6 * i + 4] = 0;
        intrinsics[6 * i + 5] = 0;
    }

    const int origin_frame = 0; // One should be able to choose if they know the optimal frame
    if (this->initGroundTruth) {
        for (int i = 0; i < this->getNumFrames(); ++i) {
            auto traj = this->sensor->GetTrajectory(i);

            Eigen::Matrix3f rot = traj.block(0, 0, 3, 3);
            auto r = Eigen::AngleAxis<float>(rot);

            R[3 * i + 0] = r.axis()(0) * r.angle();
            R[3 * i + 1] = r.axis()(1) * r.angle();
            R[3 * i + 2] = r.axis()(2) * r.angle();
            T[3 * i + 0] = traj(0, 3);
            T[3 * i + 1] = traj(1, 3);
            T[3 * i + 2] = traj(2, 3);

            fmt::print("\nTrajectory at frame: {}\n", i);
            std::cout << traj << std::endl;
            fmt::print("Axis: {} {} {}\n", r.axis()(0), r.axis()(1), r.axis()(2));
            fmt::print("Angle: {}\n", r.angle());


            setEstimatedPose(i, traj);
        }
    } else {
        std::vector<Eigen::Vector3f> reference_points;
        std::vector<int> reference_points_indices;

    // Get all the key points from the origin frame coords (x,y,z) and point index

    for (int obsIndex : correspondenceFinder->getCamObs(origin_frame)) {
        int pointIndex = correspondenceFinder->getObsPoint(obsIndex);

        if(pointIndex == -1) { continue; }

        Eigen::Vector3f p;
        p << x[obsIndex], y[obsIndex], z[obsIndex];
        reference_points.push_back(p);
        reference_points_indices.push_back(pointIndex);
    }

        ProcrustesAligner aligner;
        for (int frameId = 0; frameId < this->getNumFrames(); frameId++) {
            if (frameId == origin_frame) {
                std::cout << frameId << ":  " << " The origin frame" << std::endl;
                setEstimatedPose(frameId, Eigen::Matrix4f::Identity());
            } else {
                std::vector<Eigen::Vector3f> source_points;
                std::vector<int> source_points_indices;

            // todo compare to prev frame

            // get all the key points from the target frame
            for (int obsIndex : correspondenceFinder->getCamObs(frameId)) {
                int pointIndex = correspondenceFinder->getObsPoint(obsIndex);

                if(pointIndex == -1) { continue; }

                Eigen::Vector3f p;
                p << x[obsIndex], y[obsIndex], z[obsIndex];
                source_points.push_back(p);
                source_points_indices.push_back(pointIndex);
            }

            // remove all the key points that aren't in both vectors and exclude points with negative infinity depth
            std::vector<Eigen::Vector3f> matching_reference_points;
            std::vector<Eigen::Vector3f> matching_source_points;
            for(int j=0; j<reference_points.size(); j++) {
                for(int k=0; k<source_points.size(); k++) {
                    if(reference_points_indices[j] == source_points_indices[k]) {
                        if(std::isinf(reference_points[j](2))) break;
                        if(std::isinf(source_points[k](2))) break;
                        if(std::isnan(reference_points[j](2))) break;
                        if(std::isnan(source_points[k](2))) break;

                        matching_reference_points.push_back(reference_points[j]);
                        matching_source_points.push_back(source_points[k]);

                        //std::cout << reference_points[j](0) << " " << reference_points[j](1) << " " << reference_points[j](2) << "\n\n";

                        break;
                    }
                }
            }

            // for(int i=0; i++<source_points_indices.size(); i++) {
            //     std::cout << "Index:  " << source_points_indices[i] << std::endl;
            // }

            //std::cout << frameId << ":  " << matching_source_points.size() << " points" << std::endl;

                // estimatedPose transforms source -> reference
                Eigen::Matrix4f estimatedPose = aligner.estimatePose(matching_source_points, matching_reference_points);

                // visualize match
/*

                std::vector<Eigen::Vector3f> points;
                std::vector<Eigen::Vector3i> colors;
                Eigen::Vector3i red;
                red << 255,0,0;
                Eigen::Vector3i green;
                green << 0, 255, 0;
                Eigen::Vector3i blue;
                green << 0,0, 255;
                for(auto &point : matching_reference_points) {
                    points.push_back(point);
                    colors.push_back(red);
                }
                for(auto &point : matching_source_points) {
                    points.push_back(point);
                    colors.push_back(blue);
                }
                for(auto &point : matching_source_points) {
                    Eigen::Vector4f p;
                    p << point, 1;
                    auto res = (estimatedPose*p) ;
                    points.push_back((res.head<3>().array()+ 0.00001f).matrix());
                    colors.push_back(green);
                }

                MeshWriter::WriteToPLYFile(fmt::format("procrustes_0_{}.ply", frameId), points, colors);
*/

                setEstimatedPose(frameId, estimatedPose);
            }
        }
        for (int frameId = 0; frameId < this->getNumFrames(); ++frameId) {

            Eigen::Matrix4f pose = Eigen::Map<const Eigen::Matrix4f>(getEstimatedPose(frameId));

        Eigen::Matrix3f rotation_matrix;
        rotation_matrix << pose.block(0, 0, 3, 3);
        Eigen::AngleAxis<float> r = Eigen::AngleAxis<float>(rotation_matrix);

            // todo multiply with prev camera pose
            if (frameId == origin_frame) {
                R[3 * frameId + 0] = 0;
                R[3 * frameId + 1] = 0;
                R[3 * frameId + 2] = 0;
            } else {
                R[3 * frameId + 0] = r.axis()(0) * r.angle();
                R[3 * frameId + 1] = r.axis()(1) * r.angle();
                R[3 * frameId + 2] = r.axis()(2) * r.angle();
            }
            T[3 * frameId + 0] = pose(0, 3);
            T[3 * frameId + 1] = pose(1, 3);
            T[3 * frameId + 2] = pose(2, 3);
        }

    }
    fmt::print("\n\nInitializing 3d points\n");

    for (int i = 0; i < this->getNumPoints(); ++i) {
        auto observationsIds = correspondenceFinder->getPointObs(i);
        Eigen::Matrix4f pose;
        pose.setZero();
        Eigen::Vector4f point;
        bool foundValidObsDepth = false;

        for(int obsIndex : observationsIds) {
            int frameId = getObsCam(obsIndex);
            pose = Eigen::Map<const Eigen::Matrix4f>(getEstimatedPose(frameId));
            point << x[obsIndex], y[obsIndex], z[obsIndex], 1;

            fmt::print("Point {0} with obs {1} in frame {2}: {3:.4f} {4:.4f} {5:.4f} {6:.4f}\n", i, obsIndex, frameId, point(0), point(1), point(2), point(3));
            std::cout << pose << std::endl;

            // if value is inf or nan then invalidate obs_point
            if (!std::isinf(point(0)) && !std::isinf(point(1)) && !std::isinf(point(2)) &&
                !std::isnan(point(0)) && !std::isnan(point(1)) && !std::isnan(point(2))) {
                foundValidObsDepth = true;
                break;
            } // we could leave the correspondences as long as we find its 3D position
            else {
                fmt::print("nan\n");
            }
        }

        if(!foundValidObsDepth){
            fmt::print("Invalidating 3d point {}\n", i);

            // invalidate obs_point for all observations, no 3D position known
            for(int obsIndex : observationsIds) {
                correspondenceFinder->obs_point[obsIndex] = -1;
            }

            X[3 * i + 0] = nan("");
            X[3 * i + 1] = nan("");
            X[3 * i + 2] = nan("");
        } else {

            auto point3D = pose * point;
            fmt::print("3d Point: {0:.4f} {1:.4f} {2:.4f}\n\n", point3D(0), point3D(1), point3D(2));

            X[3 * i + 0] = point3D(0);
            X[3 * i + 1] = point3D(1);
            X[3 * i + 2] = point3D(2);
        }
    }
}


Eigen::Vector3i KinectDataloader::getPointColor(int point_index) const {
    Eigen::Vector3i rgb_vector;
    rgb_vector << 0, 0, 0;

    auto observationsIds = correspondenceFinder->getPointObs(point_index);

    for (int obs_idx : observationsIds) {
        int cam_idx = getObsCam(obs_idx);
        auto frame_width = getColor(cam_idx).size[1];
        auto frame_height = getColor(cam_idx).size[0];
        auto point = correspondenceFinder->getObservation(obs_idx);

        assert(point.x <= frame_width);
        assert(point.y <= frame_height);

        if (std::isnan(point.x) || std::isnan(point.y)) {
            continue;
        }
        cv::Vec3b color = colorImages[cam_idx].at<cv::Vec3b>(point.y, point.x);
        uint b = color[0];
        uint g = color[1];
        uint r = color[2];
        rgb_vector << r, g, b;
        return rgb_vector;
    }

    return rgb_vector;
}

const float* KinectDataloader::getEstimatedPose(int i) const {
    return &estimatedPoses[16*i];
}
void KinectDataloader::setEstimatedPose(int i, Eigen::Matrix4f mat){
    auto* pose = &estimatedPoses[16*i];
// column major
    pose[0] = mat(0,0);
    pose[1] = mat(1,0);
    pose[2] = mat(2,0);
    pose[3] = mat(3,0);
    pose[4] = mat(0,1);
    pose[5] = mat(1,1);
    pose[6] = mat(2,1);
    pose[7] = mat(3,1);
    pose[8] = mat( 0,2);
    pose[9] = mat( 1,2);
    pose[10] = mat(2,2);
    pose[11] = mat(3,2);
    pose[12] = mat(0,3);
    pose[13] = mat(1,3);
    pose[14] = mat(2,3);
    pose[15] = mat(3,3);

}




