
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <tuple>

#include "bundleadjust/PointMatching.h"
#include "bundleadjust/ProcrustesAligner.h"

#include "bundleadjust/util.h"

using namespace cv;
using namespace cv::xfeatures2d;




OnlinePointMatcher::OnlinePointMatcher(const Ptr<cv::FeatureDetector> detector,
                                       const Ptr<cv::DescriptorExtractor> extractor,
                                       const Ptr<cv::DescriptorMatcher> matcher,
                                       std::unordered_map<std::string, float> params) :
        detector(detector),
        extractor(extractor),
        matcher(matcher),
        params(params) {}

void OnlinePointMatcher::extractKeypoints(const cv::Mat currentFrame) {

    // Detect keypoints
    std::vector<KeyPoint> current_frame_keypoints;
    Mat current_frame_descriptors;
    detector->detect(currentFrame, current_frame_keypoints);
    extractor->compute(currentFrame, current_frame_keypoints, current_frame_descriptors);

    this->keypoints.push_back(current_frame_keypoints);
    this->descriptors.push_back(current_frame_descriptors);

}

void OnlinePointMatcher::matchKeypoints(std::vector<cv::Mat> & depthImages, Eigen::Matrix3f & instrinsicsInv) {
    std::cout << "Matching points" << std::endl;

    const float ratio_thresh = params["ratioThreshold"];

    int num_frames = this->keypoints.size();
    int totalPointsUntilFrame[num_frames];
    int num_observations = 0;
    for (size_t i = 0; i < num_frames; ++i) {
        totalPointsUntilFrame[i] = num_observations; // excluding current frame
        auto &kps = this->keypoints[i];
        auto num_current_points = kps.size();

        const int frame_width = depthImages[i].size[0];
        const int frame_height = depthImages[i].size[1];

        // build x, y, z observations
        for(int j=0; j<num_current_points; j++) {
            double x_obs = kps[j].pt.x;
            double y_obs = kps[j].pt.y;

            Eigen::Vector3f image_point;
            image_point << x_obs, y_obs, 1;
            x_obs = clamp(x_obs, 0, frame_width);
            y_obs = clamp(y_obs, 0, frame_height);
            
            double depth = depthImages[i].at<double>(x_obs, y_obs);
            Eigen::Vector3f cameraLine = instrinsicsInv * image_point;
            Eigen::Vector4f cameraPoint; // Andrew: don't think homogenous will be necessary
            cameraPoint << depth * cameraLine, 1;

            x.push_back(cameraPoint(0));
            y.push_back(cameraPoint(1));
            z.push_back(depth);
        }
        num_observations += num_current_points;
    }

    // init index vectors
    this->obs_point.resize(num_observations, -1);
    this->obs_cam.resize(num_observations, -1);
    for (size_t i = 0; i < num_frames; ++i) {
        auto len = ((i == num_frames - 1) ? num_observations : totalPointsUntilFrame[i + 1]) - totalPointsUntilFrame[i];
        std::fill_n(&this->obs_cam[totalPointsUntilFrame[i]], len, i);
    }


    for (int frameId = 0; frameId < num_frames; ++frameId) {

        // init matcher
        matcher->clear();
        matcher->add(descriptors[frameId]); // add current descriptor to the matcher
        matcher->train(); // train matcher on descriptors

        for (int otherFrameId = 0; otherFrameId < frameId; ++otherFrameId) {
            auto &desc = descriptors[otherFrameId];

            std::vector<std::vector<cv::DMatch>> knn_matches; // mask not supported for flann
            matcher->knnMatch(desc, knn_matches, 2);

            for (int i = 0; i < knn_matches.size(); ++i) {
                std::vector<DMatch> &match = knn_matches[i];
                
                // ratio test
                if (match[0].distance < ratio_thresh * match[1].distance) {

                    auto &obs = match[0];
                    // keep track of 3D points (1 3D point corresponding to all 2D matches)
                    int *other3D = &obs_point[totalPointsUntilFrame[otherFrameId] + obs.queryIdx];
                    int *current3D = &obs_point[totalPointsUntilFrame[frameId] + obs.trainIdx];
                    if (*other3D == -1) { // 2d observation doesn't correspond to 3D point yet
                        int newPoint = this->numPoints3d++;
                        *other3D = newPoint;
                        *current3D = newPoint;
                    } else { // 2D point has already been matched to 3D point, assign new 2D point to it as well
                        *current3D = *other3D;
                    }
                }
            }
        }
    }

    std::cout << numPoints3d << std::endl;
}

std::vector<cv::Point2f> OnlinePointMatcher::getObservations() const {
    std::vector<cv::Point2f> points;

    for (auto &frame : this->keypoints) {
        for (auto &keypoint : frame) {
            points.push_back(keypoint.pt);
        }
    }

    return points;
}

// std::vector<std::tuple<cv::Point2f, cv::Point2f>> OnlinePointMatcher::get_matching_observations_between_frames(const int base_frame, const int other_frame) const {
//     std::vector<std::tuple<cv::Point2f, cv::Point2f>> points;

//     for (auto &frame : this->keypoints) {
//         for (auto &keypoint : frame) {
//             points.push_back(keypoint.pt);
//         }
//     }

//     return points;
// }

int OnlinePointMatcher::getObsCam(int index) const {
    return this->obs_cam[index];
}

int OnlinePointMatcher::getObsPoint(int index) const {
    return this->obs_point[index];
}

int OnlinePointMatcher::getNumObservations() const {
    return this->obs_point.size();
}

int OnlinePointMatcher::getNumFrames() const {
    return this->keypoints.size();
}

int OnlinePointMatcher::getNumPoints() const {
    return this->numPoints3d;
}
