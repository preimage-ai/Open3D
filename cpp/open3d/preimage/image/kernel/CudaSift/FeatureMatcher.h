#pragma once
/*******************************************************
 * Copyright (C) 2020 Preimage siddharth@preimage.ai
 *
 * This file is part of Preimage.
 *
 * Preimage can not be copied and/or distributed without the express
 * permission of the authors
 *******************************************************/
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
// #include "json.hpp"

#include "open3d/core/Tensor.h"
#include "open3d/preimage/image/kernel/CudaSift/cudaImage.h"
#include "open3d/preimage/image/kernel/CudaSift/cudaSift.h"

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

struct CamIntrinsics {
    double fx, fy, cx, cy, k1, k2, k3, p1, p2;
};

class FeatureMatcher {
public:
    /// \brief Match features between pairs of images.
    ///
    /// \param image_id_to_cam_params: map from image_id to camera parameters,
    /// where camera parameters are [fx, fy, cx, cy, k1, k2, p1, p2, k3].
    FeatureMatcher(
            const std::vector<int>& image_ids,
            const std::vector<std::pair<int, int>>& match_pairs,
            const std::unordered_map<int, std::vector<double>>&
                    image_id_to_cam_params,
            const std::unordered_map<int, std::string>& path_to_keypoints_files,
            const std::string& path_to_output_eg_file,
            const std::string& path_to_output_trackinfo_file,
            const double desc_thres = 0.85,
            const double max_ratio = 0.95,
            const double desc_thres_guided = 0.85,
            const double max_ratio_guided = 0.95,
            const bool undistort = true,
            const int device_id = 0);

    ~FeatureMatcher();

    void Run();

private:
    std::vector<int> image_ids_;
    std::vector<std::pair<int, int>> match_pairs_;
    std::unordered_map<int, int> image_id_to_index_;
    std::unordered_map<int, std::vector<double>> image_id_to_cam_params_;
    std::unordered_map<int, std::string> path_to_keypoints_files_;
    std::string path_to_output_eg_file_;
    std::string path_to_output_trackinfo_file_;
    double desc_thres_;
    double max_ratio_;
    double desc_thres_guided_;
    double max_ratio_guided_;
    bool undistort_;
    int device_id_;

    std::unordered_map<int, int> num_features_map_;
    std::unordered_map<int, SiftData> s_kpts_;

    void LoadKeypointsFromBIN();

    int ImproveHomography(SiftData& data,
                          float* homography,
                          int numLoops,
                          float minScore,
                          float maxAmbiguity,
                          float thresh);
};

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
