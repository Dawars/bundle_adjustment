#include "bundleadjust/PointMatching.h"


std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

void OnlinePointMatcher::configure_matcher(const Ptr<FeatureDetector> detector, 
                                           const Ptr<DescriptorExtractor> extractor, 
                                           const Ptr<DescriptorMatcher> matcher) {

    this->detector = detector;
    this->extractor = extractor;
    this->matcher = matcher;
}

void OnlinePointMatcher::read_images(const std::string dir) {
    glob(dir, image_paths, false);
    size_t count = image_paths.size();
    for (size_t i=0; i<count; i++) {
        images.push_back(imread(image_paths[i]));
    }
}

void OnlinePointMatcher::match_with_frame(const int frame_idx) {
    current_frame = images[frame_idx];

    for(size_t i=0; i<frame_idx; i++) {
        std::vector<Mat>::const_iterator first_frame = images.begin();
        std::vector<Mat>::const_iterator most_recent_frame = images.begin() + frame_idx - 1;
        
        std::vector<Mat> previous_frames(first_frame, most_recent_frame);
        std::vector<std::vector<std::vector<DMatch>>> matching_pairs;

        // Detect keypoints
        detector = SIFT::create(); // TODO: Make configurable
        std::vector<KeyPoint> current_frame_keypoints;
        std::vector<std::vector<KeyPoint>> previous_frames_keypoints;
        detector->detect(current_frame, current_frame_keypoints);
        detector->detect(previous_frames, previous_frames_keypoints);

        // Compute descriptors
        Mat current_frame_descriptors;
        std::vector<Mat> previous_frames_descriptors;
        extractor = SIFT::create(); // TODO: Make configurable
        extractor->compute(current_frame, current_frame_keypoints, current_frame_descriptors);
        extractor->compute(previous_frames, previous_frames_keypoints, previous_frames_descriptors);

        // Match points 
        matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); // TODO: Make configurable
        matches.resize(0);
        matcher->add(previous_frames_descriptors);
        matcher->train();
        matcher->match(current_frame_descriptors, matches);

        // TODO: Filter for good points
        // const float ratio_thresh = 0.7f;
        // std::vector<DMatch> good_matches;
        // for (size_t i = 0; i < matches.size(); i++)
        // {
        //     if (false)
        //     {
        //         good_matches.push_back(matches[i]);
        //     }
        // }

        // TODO: Rewrite for saving
        // if (i == frame_idx - 1) {
            
        //     std::vector<std::string>::const_iterator first_name = image_paths.begin();
        //     std::vector<std::string>::const_iterator last_name = image_paths.begin() + frame_idx - 1;
        //     std::vector<std::string> previous_frames_names(first_name, last_name);
            

        //     save_images_of_matches(current_frame, current_frame_keypoints, previous_frames, previous_frames_keypoints, matches, previous_frames_names, "/home/andrew/repos/tum/bundle_adjustment/output");
        // }
    }
}


void OnlinePointMatcher::save_images_of_matches(const Mat & current_frame, const std::vector<KeyPoint> & current_frame_keypoints, const std::vector<Mat> & previous_frames, const std::vector<std::vector<KeyPoint>> & previous_frames_keypoints, const std::vector<DMatch>& matches, const std::vector<std::string> & previous_frames_names, const std::string & output_dir) {
    Mat drawImg;
    std::vector<char> mask;
    for( size_t i = 0; i < previous_frames.size(); i++) {
        std::cout << "her" << "\n";
        if(!previous_frames[i].empty()) {
            std::cout << "in" << "\n";
            mask_matches_by_train_img_idx(matches, (int)i, mask );
            drawMatches(current_frame, current_frame_keypoints, previous_frames[i], previous_frames_keypoints[i], matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask);
            auto split_path = split(previous_frames_names[i], "/");
            std::string filename = output_dir + "/res_" + split_path[split_path.size()-1];
            imwrite(filename, drawImg);
        }
    }
}

void OnlinePointMatcher::mask_matches_by_train_img_idx(const std::vector<DMatch> & matches, int train_img_idx, std::vector<char> & mask) {
    mask.resize(matches.size());
    fill(mask.begin(), mask.end(), 0);

    for(size_t i = 0; i < matches.size(); i++) {
        if(matches[i].imgIdx == train_img_idx)
            mask[i] = 1;
    }
}

