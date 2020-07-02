#pragma once

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


class OnlinePointMatcher {
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorMatcher> matcher;
    Ptr<DescriptorExtractor> descriptorExtractor;

public:
    std::vector<Mat> images;
    std::vector<DMatch> matches;
    Mat current_frame;
    std::vector<std::string> image_paths;


    void read_images(const std::string dir);

    void match_with_frame(const int frame_idx);

    void save_images_of_matches(const Mat & current_frame, 
                                const std::vector<KeyPoint> & current_frame_keypoints, 
                                const std::vector<Mat> & previous_frames, 
                                const std::vector<std::vector<KeyPoint>> & previous_frames_keypoints, 
                                const std::vector<DMatch>& matches, 
                                const std::vector<std::string> & previous_frames_names, 
                                const std::string & output_dir);

    void mask_matches_by_train_img_idx(const std::vector<DMatch> & matches, int train_img_idx, std::vector<char> & mask);
};