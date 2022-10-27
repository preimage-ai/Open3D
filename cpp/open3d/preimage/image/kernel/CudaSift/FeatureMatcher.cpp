#include "open3d/preimage/image/kernel/CudaSift/FeatureMatcher.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

#include "open3d/preimage/image/kernel/CudaSift/cudaSiftH.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/json.h"

using json = nlohmann::json;

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

bool plot_matches = false;

FeatureMatcher::FeatureMatcher(
        const std::vector<int> &image_ids,
        const std::vector<std::pair<int, int>> &match_pairs,
        const std::unordered_map<int, std::vector<double>>
                &image_id_to_cam_params,
        const std::unordered_map<int, std::string> &path_to_keypoints_files,
        const std::string &path_to_output_eg_file,
        const std::string &path_to_output_trackinfo_file,
        const double desc_thres,
        const double max_ratio,
        const double desc_thres_guided,
        const double max_ratio_guided,
        const bool undistort,
        const int device_id)
    : image_ids_(image_ids),
      match_pairs_(match_pairs),
      image_id_to_cam_params_(image_id_to_cam_params),
      path_to_keypoints_files_(path_to_keypoints_files),
      path_to_output_eg_file_(path_to_output_eg_file),
      path_to_output_trackinfo_file_(path_to_output_trackinfo_file),
      desc_thres_(desc_thres),
      max_ratio_(max_ratio),
      desc_thres_guided_(desc_thres_guided),
      max_ratio_guided_(max_ratio_guided),
      undistort_(undistort),
      device_id_(device_id) {
    InitCuda(device_id_);

    std::ofstream EGfile(path_to_output_eg_file_);
    EGfile.close();
    std::ofstream TrackInfoFile(path_to_output_trackinfo_file_);
    TrackInfoFile.close();
}

FeatureMatcher::~FeatureMatcher() {}

static double AverageTriangulationAngle(cv::Mat &local_R,
                                        cv::Mat &local_t,
                                        cv::Mat &K,
                                        std::vector<cv::Point2d> &pts1,
                                        std::vector<cv::Point2d> &pts2,
                                        cv::Mat &mask) {
    cv::Mat P1, P2;
    std::vector<cv::Point2d> pts1_ref, pts2_ref;
    cv::Mat pts3D;
    cv::Mat Rt;
    double angle_sum = 0;
    cv::hconcat(local_R, local_t, Rt);
    cv::hconcat(K, cv::Mat::zeros(3, 1, CV_64F), P1);
    P2 = K * Rt;

    for (int i = 0; i < static_cast<int>(pts1.size()); i++) {
        if ((unsigned int)mask.at<uchar>(i)) {
            pts1_ref.push_back(pts1[i]);
            pts2_ref.push_back(pts2[i]);
        }
    }

    cv::triangulatePoints(P1, P2, pts1_ref, pts2_ref, pts3D);
    cv::Mat cam1_center = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat cam2_center = -local_R.t() * local_t;
    for (int it = 0; it < pts3D.cols; it++) {
        double x = pts3D.at<double>(0, it) / pts3D.at<double>(3, it);
        double y = pts3D.at<double>(1, it) / pts3D.at<double>(3, it);
        double z = pts3D.at<double>(2, it) / pts3D.at<double>(3, it);
        double a_squared = x * x + y * y + z * z;
        double b_squared = (x - cam2_center.at<double>(0, 0)) *
                                   (x - cam2_center.at<double>(0, 0)) +
                           (y - cam2_center.at<double>(1, 0)) *
                                   (y - cam2_center.at<double>(1, 0)) +
                           (z - cam2_center.at<double>(2, 0)) *
                                   (z - cam2_center.at<double>(2, 0));
        double c_squared = 1;
        double num = c_squared - a_squared - b_squared;
        double den = 2 * std::sqrt(a_squared * b_squared);
        double theta = (den == 0.0) ? 0.0 : std::abs(std::acos(num / den));
        angle_sum += std::min(theta, M_PI - theta);
    }
    return (angle_sum * 180.0 / M_PI / pts3D.cols);
}

static void UndistortKpts(uint num,
                          float *kpts,
                          double K_arr[3][3],
                          cv::Vec<double, 5> dist_coeff) {
    std::vector<cv::Point2d> pts, pts_undistorted;
    for (uint i = 0; i < num; i++) {
        pts.push_back(cv::Point2d(kpts[i * 144], kpts[i * 144 + 1]));
    }

    cv::Mat K = cv::Mat(3, 3, CV_64F, K_arr);
    cv::undistortPoints(pts, pts_undistorted, K, dist_coeff, cv::noArray(), K);

#pragma omp parallel for
    for (uint i = 0; i < num; i++) {
        kpts[i * 144] = pts_undistorted[i].x;
        kpts[i * 144 + 1] = pts_undistorted[i].y;
        // kpts[i * 144] = pts[i].x;
        // kpts[i * 144 + 1] = pts[i].y;
    }
}

static std::string logEG(int id1,
                         int id2,
                         cv::Mat local_R,
                         cv::Mat local_t,
                         int inliers,
                         double angle) {
    std::stringstream EG;
    EG << id1 << " " << id2 << " " << local_R.at<double>(0, 0) << " "
       << local_R.at<double>(0, 1) << " " << local_R.at<double>(0, 2) << " "
       << local_R.at<double>(1, 0) << " " << local_R.at<double>(1, 1) << " "
       << local_R.at<double>(1, 2) << " " << local_R.at<double>(2, 0) << " "
       << local_R.at<double>(2, 1) << " " << local_R.at<double>(2, 2) << " "
       << local_t.at<double>(0, 0) << " " << local_t.at<double>(0, 1) << " "
       << local_t.at<double>(0, 2) << " " << inliers << " " << angle;
    return EG.str();
}

// static std::string logF(int id1, int id2, cv::Mat est_F, int inliers) {
//     std::stringstream F;
//     F << id1 << " " << id2 << " " << est_F.at<double>(0, 0) << " "
//       << est_F.at<double>(0, 1) << " " << est_F.at<double>(0, 2) << " "
//       << est_F.at<double>(1, 0) << " " << est_F.at<double>(1, 1) << " "
//       << est_F.at<double>(1, 2) << " " << est_F.at<double>(2, 0) << " "
//       << est_F.at<double>(2, 1) << " " << est_F.at<double>(2, 2) << " "
//       << inliers << std::endl;
//     return F.str();
// }

static std::string rtrim(const std::string &s) {
    const std::string WHITESPACE = " \n\r\t\f\v";
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

static std::string GetTrackInfo(int id1, int id2, const SiftData &s) {
    std::stringstream trackinfo;
    trackinfo << id1 << " " << id2 << " ";
    for (int it = 0; it < s.numPts; it++) {
        if (!s.h_data[it].match_error)
            trackinfo << it << " " << s.h_data[it].match << " ";
    }
    return rtrim(trackinfo.str());
}

int FeatureMatcher::ImproveHomography(SiftData &data,
                                      float *homography,
                                      int numLoops,
                                      float minScore,
                                      float maxAmbiguity,
                                      float thresh) {
#ifdef MANAGEDMEM
    SiftPoint *mpts = data.m_data;
#else
    if (data.h_data == NULL) return 0;
    SiftPoint *mpts = data.h_data;
#endif
    float limit = thresh * thresh;
    int numPts = data.numPts;
    cv::Mat M(8, 8, CV_64FC1);
    cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
    double Y[8];
    for (int i = 0; i < 8; i++)
        A.at<double>(i, 0) = homography[i] / homography[8];
    for (int loop = 0; loop < numLoops; loop++) {
        M = cv::Scalar(0.0);
        X = cv::Scalar(0.0);
        for (int i = 0; i < numPts; i++) {
            SiftPoint &pt = mpts[i];
            if (pt.score < minScore || pt.ambiguity > maxAmbiguity) continue;
            float den = A.at<double>(6) * pt.xpos + A.at<double>(7) * pt.ypos +
                        1.0f;
            float dx = (A.at<double>(0) * pt.xpos + A.at<double>(1) * pt.ypos +
                        A.at<double>(2)) /
                               den -
                       pt.match_xpos;
            float dy = (A.at<double>(3) * pt.xpos + A.at<double>(4) * pt.ypos +
                        A.at<double>(5)) /
                               den -
                       pt.match_ypos;
            float err = dx * dx + dy * dy;
            float wei = (err < limit ? 1.0f : 0.0f);  // limit / (err + limit);
            Y[0] = pt.xpos;
            Y[1] = pt.ypos;
            Y[2] = 1.0;
            Y[3] = Y[4] = Y[5] = 0.0;
            Y[6] = -pt.xpos * pt.match_xpos;
            Y[7] = -pt.ypos * pt.match_xpos;
            for (int c = 0; c < 8; c++)
                for (int r = 0; r < 8; r++)
                    M.at<double>(r, c) += (Y[c] * Y[r] * wei);
            X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_xpos * wei);
            Y[0] = Y[1] = Y[2] = 0.0;
            Y[3] = pt.xpos;
            Y[4] = pt.ypos;
            Y[5] = 1.0;
            Y[6] = -pt.xpos * pt.match_ypos;
            Y[7] = -pt.ypos * pt.match_ypos;
            for (int c = 0; c < 8; c++)
                for (int r = 0; r < 8; r++)
                    M.at<double>(r, c) += (Y[c] * Y[r] * wei);
            X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_ypos * wei);
        }
        cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
    }
    int numfit = 0;
    for (int i = 0; i < numPts; i++) {
        SiftPoint &pt = mpts[i];
        float den = A.at<double>(6) * pt.xpos + A.at<double>(7) * pt.ypos + 1.0;
        float dx = (A.at<double>(0) * pt.xpos + A.at<double>(1) * pt.ypos +
                    A.at<double>(2)) /
                           den -
                   pt.match_xpos;
        float dy = (A.at<double>(3) * pt.xpos + A.at<double>(4) * pt.ypos +
                    A.at<double>(5)) /
                           den -
                   pt.match_ypos;
        float err = dx * dx + dy * dy;
        if (err < limit) {
            numfit++;
            pt.match_error = 0;
        } else
            pt.match_error = 1;
    }
    for (int i = 0; i < 8; i++) homography[i] = A.at<double>(i);
    homography[8] = 1.0f;
    return numfit;
    // return 0;
}

void FeatureMatcher::LoadKeypointsFromBIN() {
    // TODO: separate reading features, undist, filling values, such that
    // filling values can be done efficiently in parallel.
    for (auto k : image_ids_) {
        std::stringstream ss;
        ss << path_to_keypoints_files_[k];
        std::ifstream rf(ss.str(), std::ios::out | std::ios::binary);
        rf.seekg(0, std::ios::end);
        size_t length = rf.tellg();
        rf.seekg(0, std::ios::beg);
        const uint num_features = length / (144 * sizeof(float));
        num_features_map_[k] = num_features;
        utility::LogDebug("Loading {} keypoints from file: {}", num_features,
                          ss.str());

        float *kpts_ = (float *)malloc(int(num_features) * sizeof(float) * 144);
        FILE *file = fopen(ss.str().c_str(), "rb");
        if (fread(kpts_, sizeof(float), int(num_features) * 144, file) < 1) {
            // TODO: Handle case with skipping this image.
            open3d::utility::LogError("Read BIN failed: unexpected EOF : {}",
                                      ss.str());
        }

        if (undistort_) {
            // cam_params: h, w, fx, fy, cx, cy, k1, k2, p1, p2, k3
            const std::vector<double> cam_params = image_id_to_cam_params_[k];
            double K_arr[3][3] = {{cam_params[2], 0, cam_params[4]},
                                  {0, cam_params[3], cam_params[5]},
                                  {0, 0, 1}};
            cv::Vec<double, 5> dist_coeff(cam_params[6], cam_params[7],
                                          cam_params[8], cam_params[9],
                                          cam_params[10]);
            UndistortKpts(num_features, kpts_, K_arr, dist_coeff);
        }

        SiftData sd;
        InitSiftData(sd, num_features, true, true);
        sd.numPts = num_features;
        for (uint j = 0; j < num_features; j++) {
            sd.h_data[j].xpos = kpts_[j * 144];
            sd.h_data[j].ypos = kpts_[j * 144 + 1];
            sd.h_data[j].scale = kpts_[j * 144 + 2];
            sd.h_data[j].orientation = kpts_[j * 144 + 3];
            sd.h_data[j].sharpness = kpts_[j * 144 + 4];
            sd.h_data[j].edgeness = kpts_[j * 144 + 5];
            sd.h_data[j].ambiguity = kpts_[j * 144 + 6];
            sd.h_data[j].score = kpts_[j * 144 + 7];
            sd.h_data[j].match = -9999;
            sd.h_data[j].match_xpos = kpts_[j * 144 + 9];
            sd.h_data[j].match_ypos = kpts_[j * 144 + 10];
            sd.h_data[j].match_error = kpts_[j * 144 + 11];
            sd.h_data[j].subsampling = kpts_[j * 144 + 12];
            sd.h_data[j].empty[0] = kpts_[j * 144 + 13];
            sd.h_data[j].empty[1] = kpts_[j * 144 + 14];
            sd.h_data[j].empty[2] = kpts_[j * 144 + 15];
            std::copy(kpts_ + j * 144 + 16, kpts_ + j * 144 + 144,
                      sd.h_data[j].data);
        }
        s_kpts_.insert(std::pair<int, SiftData>(k, sd));

        // TODO: Uncomment the following like and add CopySift function.
        CopySift(sd);
        free(kpts_);
        fclose(file);
    }
}

void FeatureMatcher::Run() {
    LoadKeypointsFromBIN();

    std::ofstream EGfile(path_to_output_eg_file_);
    std::ofstream TrackInfoFile(path_to_output_trackinfo_file_,
                                std::ofstream::out | std::ofstream::app);

    json EGs_unfiltered, trackinfo_json;
    std::vector<std::string> EGs, trackinfo_list;
    for (uint it = 0; it < match_pairs_.size(); it++) {
        // std::cout << it * 100.0 / (double)match_pairs_.size() << std::endl;

        int id1 = match_pairs_[it].first;
        int id2 = match_pairs_[it].second;
        if (id1 > id2) {
            int temp = id1;
            id1 = id2;
            id2 = temp;
        }

        SiftData &s1 = s_kpts_[id1];
        SiftData &s2 = s_kpts_[id2];

        utility::LogDebug("Matching {} to {}.", id1, id2);
        MatchSiftData(s1, s2);

        float homography[9];
        int numMatches;
        FindHomography(s1, homography, &numMatches, 10000, desc_thres_,
                       max_ratio_, 5.0);

        if (numMatches) {
            utility::LogDebug(
                    "\t Found {} matches, with score: {} and ambiguity {}.",
                    numMatches, s1.h_data[id1].score, s1.h_data[id1].ambiguity);
        } else {
            continue;
        }

        const int numFit = ImproveHomography(s1, homography, 20, desc_thres_,
                                             max_ratio_, 3.0);

        std::set<int> matches_2;
        for (int it = 0; it < s1.numPts; it++) {
            if (!s1.h_data[it].match_error) {
                matches_2.insert(s1.h_data[it].match);
            }
        }
        if (float(matches_2.size()) / float(numFit) < 0.8) {
            continue;
        }

        std::vector<cv::Point2d> pts1;
        std::vector<cv::Point2d> pts2;
        pts1.reserve(numFit);
        pts2.reserve(numFit);
        // double epi_thres = 2;

        for (int it = 0; it < s1.numPts; it++) {
            if (s1.h_data[it].score > desc_thres_ &&
                s1.h_data[it].ambiguity < max_ratio_) {
                pts1.push_back(
                        cv::Point2d(s1.h_data[it].xpos, s1.h_data[it].ypos));
                pts2.push_back(cv::Point2d(s1.h_data[it].match_xpos,
                                           s1.h_data[it].match_ypos));
            }
        }

        const std::vector<double> cam_params = image_id_to_cam_params_[id1];
        double K_arr[3][3] = {{cam_params[2], 0, cam_params[4]},
                              {0, cam_params[3], cam_params[5]},
                              {0, 0, 1}};
        cv::Mat K = cv::Mat(3, 3, CV_64F, K_arr);
        cv::Mat local_R, local_t;
        cv::Mat local_R2, local_t2;
        int inliers_homo = 0;
        // int inliers = 0;
        cv::Mat mask;
        cv::Mat mask2;
        // cv::Mat F;
        cv::Mat rotvec;
        try {
            // TODO: Use different overload of the findFundamentalMat function.
            cv::Mat E = cv::findEssentialMat(
                    pts1, pts2, cv::Mat::eye(3, 3, CV_64F), cv::USAC_PARALLEL,
                    0.99999999, 1.0, mask);
            inliers_homo =
                    cv::recoverPose(E, pts1, pts2, K, local_R, local_t, mask);
            // F = K.inv().t() * E * K.inv();
        } catch (...) {
            continue;
        }

        // TODO: Code is removed from here. Refer legacy pipeline.
        // Plot matches here for debugging and filter outliers.
        // F = F.t();

        //////////////////////////////
        for (int it = 0; it < s1.numPts; it++) {
            if (s1.h_data[it].ambiguity < max_ratio_guided_ &&
                s1.h_data[it].score >
                        desc_thres_guided_) {  // s1.h_data[it].ambiguity
                                               // < max_ratio_ &&
                                               // s1.h_data[it].score
                                               // > desc_thres_ &&
                s1.h_data[it].match_error = 0;
                // inliers++;
            } else
                s1.h_data[it].match_error = 1;
        }

        if (inliers_homo < 150) {
            continue;
        }

        const double tri_angle = AverageTriangulationAngle(local_R, local_t, K,
                                                           pts1, pts2, mask);
        std::string EG =
                logEG(id1, id2, local_R, local_t, inliers_homo, tri_angle);
        EGs.push_back(EG);

        std::string trackinfo = GetTrackInfo(id1, id2, s1);
        trackinfo_list.push_back(trackinfo);
    }

    EGs_unfiltered["EGs"] = EGs;
    EGfile << std::setw(4) << EGs_unfiltered;

    trackinfo_json["trackinfo"] = trackinfo_list;
    TrackInfoFile << std::setw(4) << trackinfo_json;
}

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
