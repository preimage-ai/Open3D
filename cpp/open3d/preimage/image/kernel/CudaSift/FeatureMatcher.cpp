#include "open3d/preimage/image/kernel/CudaSift/FeatureMatcher.h"

#include "open3d/utility/json.h"
#include "open3d/utility/Logging.h"

using json = nlohmann::json;

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

bool plot_matches = false;

FeatureMatcher::FeatureMatcher(std::string egfile_path,
                               std::string trackinfo_path,
                               std::string dist_params_path,
                               std::string chunk_path,
                               bool distortion) {
    egfile_path_ = egfile_path;
    ffile_path_ = egfile_path;
    trackinfo_path_ = trackinfo_path;
    std::ofstream EGfile(egfile_path_);
    EGfile.close();
    std::ofstream TrackInfoFile(trackinfo_path_);
    TrackInfoFile.close();

    std::ifstream datasrc(chunk_path);
    json j;
    datasrc >> j;
    float focalx = j["focalxs"][0];
    float focaly = j["focalys"][0];

    if (distortion) {
        std::ifstream dist_params_file(dist_params_path);
        json dist_json;
        dist_params_file >> dist_json;

        for (int it = 0;
             it < static_cast<int>(dist_json["distortion_params"].size());
             it++) {
            dist_params_.push_back(dist_json["distortion_params"][it]);
        }
    } else {
        dist_params_.push_back(focalx);
        dist_params_.push_back(focaly);

        for (int it = 0; it < 15; it++) {
            dist_params_.push_back(0.0);
        }
    }
}

void FeatureMatcher::AverageTriangulationAngle(
        cv::Mat &local_R,
        cv::Mat &local_t,
        cv::Mat &K,
        std::vector<cv::Point2d> &pts1,
        std::vector<cv::Point2d> &pts2,
        double &angle,
        cv::Mat &mask,
        bool homog_fit,
        std::vector<int> homog_match_error) {
    cv::Mat P1, P2;
    std::vector<cv::Point2d> pts1_ref, pts2_ref;
    cv::Mat pts3D;
    cv::Mat Rt;
    double angle_sum = 0;
    cv::hconcat(local_R, local_t, Rt);
    cv::hconcat(K, cv::Mat::zeros(3, 1, CV_64F), P1);
    P2 = K * Rt;

    // if (!homog_fit){
    for (int i = 0; i < static_cast<int>(pts1.size()); i++) {
        if ((unsigned int)mask.at<uchar>(i)) {
            pts1_ref.push_back(pts1[i]);
            pts2_ref.push_back(pts2[i]);
        }
    }
    // }
    // else{
    //   for(int i = 0; i < pts1.size(); i++){
    //     // if (homog_match_error[i]){
    //       pts1_ref.push_back(pts1[i]);
    //       pts2_ref.push_back(pts2[i]);
    //     // }
    //   }
    // }

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
    angle = angle_sum * 180.0 / M_PI / pts3D.cols;
}

void FeatureMatcher::undistortKpts(uint num,
                                   float *kpts,
                                   double K_arr[3][3],
                                   cv::Vec<double, 5> dist_coeff) {
    std::vector<cv::Point2d> pts, pts_undistorted;
    for (uint i = 0; i < num; i++) {
        pts.push_back(cv::Point2d(kpts[i * 144], kpts[i * 144 + 1]));
    }
    cv::Mat K = cv::Mat(3, 3, CV_64F, K_arr);
    cv::undistortPoints(pts, pts_undistorted, K, dist_coeff, cv::noArray(), K);
    for (uint i = 0; i < num; i++) {
        kpts[i * 144] = pts_undistorted[i].x;
        kpts[i * 144 + 1] = pts_undistorted[i].y;
        // kpts[i * 144] = pts[i].x;
        // kpts[i * 144 + 1] = pts[i].y;
    }
}

std::string FeatureMatcher::logEG(int id1,
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
    // EGfile.close();
    // file_lock_.unlock();
    return EG.str();
}

void FeatureMatcher::logF(int id1, int id2, cv::Mat est_F, int inliers) {
    std::cout << "In F: " << id1 << " " << id2 << std::endl;
    file_lock_.lock();
    std::ofstream Ffile(ffile_path_, std::ofstream::out | std::ofstream::app);
    Ffile << id1 << " " << id2 << " " << est_F.at<double>(0, 0) << " "
          << est_F.at<double>(0, 1) << " " << est_F.at<double>(0, 2) << " "
          << est_F.at<double>(1, 0) << " " << est_F.at<double>(1, 1) << " "
          << est_F.at<double>(1, 2) << " " << est_F.at<double>(2, 0) << " "
          << est_F.at<double>(2, 1) << " " << est_F.at<double>(2, 2) << " "
          << inliers << std::endl;
    Ffile.close();
    file_lock_.unlock();
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
}

void FeatureMatcher::readBinFiles(const std::string folder,
                                  std::vector<int> &ids) {
    for (auto k : ids) {
        std::stringstream ss;
        ss << folder << k << ".bin";
        std::cout << ss.str() << std::endl;
        std::ifstream rf(ss.str(), std::ios::out | std::ios::binary);
        rf.seekg(0, std::ios::end);
        size_t length = rf.tellg();
        rf.seekg(0, std::ios::beg);
        uint num_features = length / (144 * sizeof(float));
        std::cout << num_features << std::endl;
        num_features_map_[k] = num_features;

        float *kpts_ = (float *)malloc(int(num_features) * sizeof(float) * 144);

        FILE *file = fopen(ss.str().c_str(), "rb");

        if (fread(kpts_, sizeof(float), int(num_features) * 144, file) < 1) {
            open3d::utility::LogWarning("Read BIN failed: unexpected EOF : {}",
                                        ss.str());
            return;
        }

        double K_arr[3][3] = {{cam_intrinsics_.fx, 0, cam_intrinsics_.cx},
                              {0, cam_intrinsics_.fy, cam_intrinsics_.cy},
                              {0, 0, 1}};
        cv::Vec<double, 5> dist_coeff(cam_intrinsics_.k1, cam_intrinsics_.k2,
                                      cam_intrinsics_.p1, cam_intrinsics_.p2,
                                      cam_intrinsics_.k3);
        // std::cout << "1" << std::endl;
        undistortKpts(num_features, kpts_, K_arr, dist_coeff);
        // std::cout << "2" << std::endl;
        SiftData sd;
        InitSiftData(sd, 32768, true, true);
        // InitSiftData(sd, num_features, true, true); 32768

        // std::cout << num_features << std::endl;
        sd.numPts = num_features;
        for (uint j = 0; j < num_features; j++) {
            // std::cout << j << std::endl;
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
        // std::cout << "0" << std::endl;
        s_kpts_.insert(std::pair<int, SiftData>(k, sd));
        // std::cout << "4" << std::endl;

        // TODO: Uncomment the following like and add CopySift function.
        // CopySift(sd);
        free(kpts_);
        fclose(file);
    }
}

void FeatureMatcher::readPairsJson(const std::string json_path) {
    InitCuda(0);
    std::cout << "Folder: " << json_path << std::endl;
    json_path_ = json_path;
    std::ifstream datasrc(json_path);
    std::string folder;
    json j;
    datasrc >> j;
    bool homog_fit = false;
    int num_matched_images, num_disjoint_images;
    try {
        num_matched_images = j["num_matched_images"];
        num_disjoint_images = j["num_disjoint_images"];
        folder = j["kpts_folder"];
    } catch (...) {
        std::cout << "Invalid File!" << std::endl;
        return;
    }
    if (static_cast<int>(j["matched_images"].size()) != num_matched_images) {
        std::cout << "Invalid File! Num Images don't match!" << std::endl;
        return;
    }
    if (static_cast<int>(j["disjoint_images"].size()) != num_disjoint_images) {
        std::cout << "Invalid File! Num Images don't match!" << std::endl;
        return;
    }
    std::vector<int> ids;
    for (auto im : j["matched_images"]) {
        ids.push_back(im);
    }

    cam_intrinsics_.fx = dist_params_[0];
    cam_intrinsics_.fy = dist_params_[1];
    cam_intrinsics_.cx = ((double)j["width"]) / 2 + dist_params_[2];
    cam_intrinsics_.cy = ((double)j["height"]) / 2 + dist_params_[3];
    cam_intrinsics_.k1 = dist_params_[4];
    cam_intrinsics_.k2 = dist_params_[5];
    cam_intrinsics_.k3 = dist_params_[6];
    cam_intrinsics_.p1 = dist_params_[7];
    cam_intrinsics_.p2 = dist_params_[8];

    std::cout << cam_intrinsics_.fx << " " << cam_intrinsics_.fy << " "
              << cam_intrinsics_.cx << " " << cam_intrinsics_.cy << " "
              << cam_intrinsics_.k1 << " " << cam_intrinsics_.k2 << std::endl;

    readBinFiles(folder, ids);
    std::vector<int> image_pairs;

    for (auto im : j["image_pairs"]) {
        image_pairs.push_back(im);
    }

    std::vector<std::string> img_paths;
    for (auto im : j["images_paths"]) {
        img_paths.push_back(im);
    }
    std::cout << image_pairs.size() << std::endl;
    std::map<int, std::string> names;

    double desc_thres = 0.85;
    double max_ratio = 0.95;

    double desc_thres_guided = 0.85;
    double max_ratio_guided = 0.95;

    std::cout << "HARDCODED PATH WARNING!!! TODO: FIX" << std::endl;
    std::string match_file_path =
            "/home/ubuntu/Datasets/small_village_2/matches.txt";
    std::ofstream MatchFile(match_file_path,
                            std::ofstream::out | std::ofstream::app);

    //   omp_set_num_threads(4);
    // #pragma omp parallel for
    // file_lock_.lock();
    std::ofstream EGfile(egfile_path_);
    std::ofstream TrackInfoFile(trackinfo_path_,
                                std::ofstream::out | std::ofstream::app);

    json EGs_unfiltered, trackinfo_json;
    std::vector<std::string> EGs, trackinfo_list;
    for (uint it = 0; it < image_pairs.size() / 2; it++) {
        std::cout << it * 200.0 / (double)image_pairs.size() << std::endl;
        int id1 = image_pairs[2 * it];
        int id2 = image_pairs[2 * it + 1];
        if (id1 > id2) {
            int temp = id1;
            id1 = id2;
            id2 = temp;
        }

        SiftData &s1 = s_kpts_[id1];
        SiftData &s2 = s_kpts_[id2];

        // std::cout << id1 << " " << id2 << std::endl;
        MatchSiftData(s1, s2);

        float homography[9];
        int numMatches;
        // // std::cout << id1 << " " << id2 << " " << s1.h_data[id1].score << "
        // " << s1.h_data[id1].ambiguity << std::endl;
        FindHomography(s1, homography, &numMatches, 10000, desc_thres,
                       max_ratio, 5.0);

        if (numMatches == 0) {
            // std::cout << "num matches" << std::endl;
            continue;
        }
        int numFit = ImproveHomography(s1, homography, 20, desc_thres,
                                       max_ratio, 3.0);

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
        double tri_angle;
        // double epi_thres = 2;

        MatchFile << id1 << " " << id2 << std::endl;
        std::vector<int> homog_match_error;

        for (int it = 0; it < s1.numPts; it++) {
            if (s1.h_data[it].score > desc_thres &&
                s1.h_data[it].ambiguity < max_ratio) {
                MatchFile << s1.h_data[it].xpos << " "
                          << s1.h_data[it].match_xpos << " "
                          << s1.h_data[it].ypos << " "
                          << s1.h_data[it].match_ypos << std::endl;
                pts1.push_back(
                        cv::Point2d(s1.h_data[it].xpos, s1.h_data[it].ypos));
                pts2.push_back(cv::Point2d(s1.h_data[it].match_xpos,
                                           s1.h_data[it].match_ypos));
            }
        }

        double K_arr[3][3] = {{cam_intrinsics_.fx, 0, cam_intrinsics_.cx},
                              {0, cam_intrinsics_.fy, cam_intrinsics_.cy},
                              {0, 0, 1}};
        cv::Mat K = cv::Mat(3, 3, CV_64F, K_arr);
        cv::Mat local_R, local_t;
        cv::Mat local_R2, local_t2;
        int inliers_homo;
        // int inliers = 0;
        cv::Mat mask;
        cv::Mat mask2;
        cv::Mat F;
        cv::Mat rotvec;
        try {
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::USAC_PARALLEL,
                                             0.99999999, 1.0, mask);
            inliers_homo =
                    cv::recoverPose(E, pts1, pts2, K, local_R, local_t, mask);
            F = K.inv().t() * E * K.inv();

        } catch (...) {
            continue;
        }

        // TODO: Code is removed from here. Refer legacy pipeline.
        // Plot matches here for debugging and filter outliers.
        F = F.t();

        //////////////////////////////
        for (int it = 0; it < s1.numPts; it++) {
            if (s1.h_data[it].ambiguity < max_ratio_guided &&
                s1.h_data[it].score >
                        desc_thres_guided) {  // s1.h_data[it].ambiguity
                                              // < max_ratio &&
                                              // s1.h_data[it].score
                                              // > desc_thres &&
                s1.h_data[it].match_error = 0;
                // inliers++;
            } else
                s1.h_data[it].match_error = 1;
        }

        if (inliers_homo < 150) {
            continue;
        }

        AverageTriangulationAngle(local_R, local_t, K, pts1, pts2, tri_angle,
                                  mask, homog_fit, homog_match_error);
        std::string EG =
                logEG(id1, id2, local_R, local_t, inliers_homo, tri_angle);
        EGs.push_back(EG);

        std::string trackinfo = saveTrackInfo(id1, id2, s1);
        trackinfo_list.push_back(trackinfo);
    }

    EGs_unfiltered["EGs"] = EGs;
    EGfile << std::setw(4) << EGs_unfiltered;

    trackinfo_json["trackinfo"] = trackinfo_list;
    TrackInfoFile << std::setw(4) << trackinfo_json;
}

void FeatureMatcher::GetEpipolarError(cv::Mat &F,
                                      cv::Vec3d &pt1,
                                      cv::Vec3d &pt2,
                                      double &ep) {
    cv::Mat err = (pt2.t() * F * pt1);
    cv::Mat line = F * pt1;
    double normalise_factor =
            std::sqrt(line.at<double>(0, 0) * line.at<double>(0, 0) +
                      line.at<double>(0, 1) * line.at<double>(0, 1));
    ep = fabs(err.at<double>(0, 0) / normalise_factor);
}

// void FeatureMatcher::freeKptsData(const json &j) {
//     for (int it = 0; it < j["matched_images"].size(); it++) {
//         int id = j["matched_images"][it];
//         FreeSiftData(s_kpts_[id]);
//     }
// }

std::string rtrim(const std::string &s) {
    const std::string WHITESPACE = " \n\r\t\f\v";
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string FeatureMatcher::saveTrackInfo(int id1, int id2, const SiftData &s) {
    // file_lock_.lock();
    // std::ofstream TrackInfoFile(trackinfo_path_, std::ofstream::out |
    // std::ofstream::app);
    std::stringstream trackinfo;
    trackinfo << id1 << " " << id2 << " ";
    for (int it = 0; it < s.numPts; it++) {
        if (!s.h_data[it].match_error)
            trackinfo << it << " " << s.h_data[it].match << " ";
    }

    return rtrim(trackinfo.str());
    // TrackInfoFile << std::endl;
    // TrackInfoFile.close();
    // file_lock_.unlock();
}

void FeatureMatcher::matchPairs() {}

FeatureMatcher::~FeatureMatcher() {
    std::ifstream datasrc(json_path_);
    json j;
    datasrc >> j;
    // freeKptsData(j);
}

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
