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
#include <opencv4/opencv2/opencv.hpp>
#include <set>
// #include "json.hpp"

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
    FeatureMatcher(std::string egfile_path,
                   std::string trackinfo_path,
                   std::string dist_params_path,
                   std::string chunk_dir,
                   bool distortion = true);
    ~FeatureMatcher();
    void readPairsJson(const std::string path);

private:
    std::map<int, int> num_features_map_;
    std::map<int, SiftData> s_kpts_;
    std::mutex file_lock_;
    std::vector<double> dist_params_;
    std::string egfile_path_;
    std::string ffile_path_;
    std::string trackinfo_path_;
    std::string coords_path_;
    std::string chunk_dir_, json_path_;
    CamIntrinsics cam_intrinsics_;
    void undistortKpts(uint num,
                       float* kpts,
                       double K_arr[3][3],
                       cv::Vec<double, 5> dist_coeff);
    // void freeKptsData(const nlohmann::json &j);
    std::string saveTrackInfo(int id1, int id2, const SiftData& s);
    void matchPairs();
    void readBinFiles(const std::string folder, std::vector<int>& ids);
    std::string logEG(int id1,
                      int id2,
                      cv::Mat local_R,
                      cv::Mat local_t,
                      int inliers,
                      double angle);
    void logF(int id1, int id2, cv::Mat F, int inliers);
    int ImproveHomography(SiftData& data,
                          float* homography,
                          int numLoops,
                          float minScore,
                          float maxAmbiguity,
                          float thresh);
    void AverageTriangulationAngle(cv::Mat& local_R,
                                   cv::Mat& local_t,
                                   cv::Mat& K,
                                   std::vector<cv::Point2d>& pts1,
                                   std::vector<cv::Point2d>& pts2,
                                   double& angle,
                                   cv::Mat& mask,
                                   bool homog_fit,
                                   std::vector<int> homog_match_error);
    void GetEpipolarError(cv::Mat& F,
                          cv::Vec3d& pt1,
                          cv::Vec3d& pt2,
                          double& ep);
};

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
