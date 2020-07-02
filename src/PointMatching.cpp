
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


#include "bundleadjust/PointMatching.h"

using namespace cv;
using namespace cv::xfeatures2d;

OnlinePointMatcher::OnlinePointMatcher() {
    detector = SIFT::create(); // TODO: Make configurable
    descriptorExtractor = SIFT::create(); // TODO: Make configurable
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); // TODO: Make configurable
}

void OnlinePointMatcher::extractKeypoints(const cv::Mat currentFrame) {

    // Detect keypoints
    std::vector<KeyPoint> current_frame_keypoints;
    Mat current_frame_descriptors;
    detector->detect(currentFrame, current_frame_keypoints);
    descriptorExtractor->compute(currentFrame, current_frame_keypoints, current_frame_descriptors);

    this->keypoints.push_back(current_frame_keypoints);
    this->descriptors.push_back(current_frame_descriptors);

    this->currentId++;
}

void OnlinePointMatcher::matchKeypoints() {

    int num_frames = this->keypoints.size();
    int totalPointsUntilFrame[num_frames];
    int num_observations = 0;
    for (size_t i = 0; i < num_frames; ++i) {
        totalPointsUntilFrame[i] = num_observations; // excluding current frame

        auto &kps = this->keypoints[i];
        auto num_current_points = kps.size();
        num_observations += num_current_points;
    }

    // init index vectors
    this->obs_point.resize(num_observations, -1);
    this->obs_cam.resize(num_observations, -1);
    for (size_t i = 0; i < num_frames; ++i) {
        auto len = ((i == num_frames - 1) ? num_observations : totalPointsUntilFrame[i + 1]) - totalPointsUntilFrame[i];
        std::fill_n(&this->obs_cam[totalPointsUntilFrame[i]], len, i);
    }

    // init matcher
    matcher->add(descriptors); // add current descriptor to the matcher
    matcher->train(); // train matcher on descriptors

    for (int frameId = 0; frameId < num_frames; ++frameId) {

        auto &desc = descriptors[frameId];

        std::vector<std::vector<cv::DMatch>> knn_matches; // mask not supported for flann
        matcher->knnMatch(desc, knn_matches, 3, true); // 3 to skip 1st match with same frame

        const float ratio_thresh = 0.7f; // todo param

        for (int i = 0; i < knn_matches.size(); ++i) {
            std::vector<DMatch> &match = knn_matches[i];

            int offset = 0;
            if (match[0].imgIdx == frameId) { // closest point is same point on same image, skip
                offset = 1;
            }

            // ratio test
            if (match[offset + 0].distance < ratio_thresh * match[offset + 1].distance) {

                auto &obs = match[offset + 0];
                // 3d points correspond to flann train point ids
                obs_point[totalPointsUntilFrame[frameId] + obs.queryIdx] = obs.trainIdx;
            }

            // TODO: RANSAC
        }
    }
}