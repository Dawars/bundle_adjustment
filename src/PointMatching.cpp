
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

void OnlinePointMatcher::matchKeypoints() {
    std::cout << "Matching points" << std::endl;

    const float ratio_thresh = params["ratioThreshold"];
    const float eps = params["ransacEps"];

    int num_frames = this->keypoints.size();
    int num_observations = 0;

    cam_obs.resize(num_frames, std::set<int>());

    for (size_t i = 0; i < num_frames; ++i) {
        totalPointsUntilFrame.push_back(num_observations); // excluding current frame

        auto &kps = this->keypoints[i];
        auto num_current_points = kps.size();

        for (int j = 0; j < keypoints[i].size(); ++j) {
            cam_obs[i].insert(totalPointsUntilFrame[i] + j);
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
            if(descriptors[frameId].rows == 0) { continue; }

            // keep statistics on number of outliers
            int all = 0;
            int filtered = 0;
            auto &desc = descriptors[otherFrameId];

            std::vector<std::vector<cv::DMatch>> knn_matches; // mask not supported for flann
            matcher->knnMatch(desc, knn_matches, 2);

            std::vector<cv::Point2f> kp_coords1;
            std::vector<cv::Point2f> kp_coords2;
            std::vector<cv::DMatch> matches;

            for (int i = 0; i < knn_matches.size(); ++i) {
                std::vector<DMatch> &match = knn_matches[i];

                // ratio test
                if (match[0].distance < ratio_thresh * match[1].distance) {
                    auto &obs = match[0];
                    // save keypoints coordinates for ransac
                    kp_coords1.push_back(keypoints[frameId][obs.trainIdx].pt);
                    kp_coords2.push_back(keypoints[otherFrameId][obs.queryIdx].pt);
                    matches.push_back(obs);
                }
            }

            // RANSAC requires at least 8 points.
            // It doesn't make a lot of sense to count the match if there are less then 8 points though
            if (kp_coords1.size() > 8) {
                // might produce empty matrix
                cv::Mat fundamental_matrix =
                        findHomography(kp_coords1, kp_coords2, cv::FM_RANSAC);
                // delete outliers
                all += kp_coords1.size();

                for (int i = 0; i < kp_coords1.size() && !fundamental_matrix.empty(); ++i) {
                    cv::Point3d p1 = {kp_coords1[i].x, kp_coords1[i].y, 1};
                    cv::Point3d p2 = {kp_coords2[i].x, kp_coords2[i].y, 1};
                    cv::Mat mp1(p1);
                    cv::Mat mp2(p2);
                    cv::Mat mp3 = fundamental_matrix * mp1;
                    cv::Point3d p3(mp3);
                    p3.x /= p3.z;
                    p3.y /= p3.z;
                    if (cv::norm(p2 - p3) < eps) {
                        filtered++;
                        auto &obs = matches[i];
                        // keep track of 3D points (1 3D point corresponding to all 2D matches)
                        int otherObsIndex = totalPointsUntilFrame[otherFrameId] + obs.queryIdx;
                        int currentObsIndex = totalPointsUntilFrame[frameId] + obs.trainIdx;
                        int *other3D = &obs_point[otherObsIndex];
                        int *current3D = &obs_point[currentObsIndex];
                        if (*other3D == -1) { // 2d observation doesn't correspond to 3D point yet
                            int newPoint = this->numPoints3d++;
                            *other3D = newPoint;
                            *current3D = newPoint;
                            point_obs.push_back(std::set<int>());
                        } else { // 2D point has already been matched to 3D point, assign new 2D point to it as well
                            *current3D = *other3D;
                        }
                        point_obs[*other3D].insert(otherObsIndex);
                        point_obs[*other3D].insert(currentObsIndex);
                    }
                }
//                std::cout << all << " " << filtered << std::endl;
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
cv::Point2f OnlinePointMatcher::getObservation(int index) const {
    for (int frameId = 0; frameId < getNumFrames(); ++frameId) {
        if(index - totalPointsUntilFrame[frameId] <= 0){
            return keypoints[frameId-1][index - totalPointsUntilFrame[frameId-1]].pt;
        }
    }
    throw std::invalid_argument("Invalid observation index");
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
int OnlinePointMatcher::getObsIndex(int frameId, int obsId){
    return totalPointsUntilFrame[frameId] + obsId;
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

std::vector<std::vector<cv::KeyPoint>> OnlinePointMatcher::getKeyPoints() const {
    return this->keypoints;
}