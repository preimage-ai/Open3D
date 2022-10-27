// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

// #include <json/json.h>
#include <vips/vips.h>

#include <fstream>
#include <iostream>

#include "open3d/Open3D.h"

using namespace open3d;

void PrintHelp() {
    using namespace open3d;

    // Currently reading images from path stored in
    // dataset_detectionn_chunk_X.json using VIPS. Later, directly read from
    // medimgs.npy.

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > PimgFM --dataset_name dataset --chunk_json_path dataset_detection_chunk_X.json --device CUDA:0");
    // clang-format on
    utility::LogInfo("");
}

std::unordered_map<int, std::vector<double>> GetCameraParamsFromInfoFiles(
        std::vector<int>& image_ids,
        std::string images_info_path,
        std::string cameras_info_path) {
    using json = nlohmann::json;

    std::ifstream images_info_stream(images_info_path);
    json images_info;
    images_info_stream >> images_info;

    std::ifstream cameras_info_stream(cameras_info_path);
    json cameras_info;
    cameras_info_stream >> cameras_info;

    std::unordered_map<int, std::vector<double>> image_id_to_camera_params;
    for (auto image_id : image_ids) {
        int cam_id = images_info[std::to_string(image_id)]["cam_id"];
        const std::string cam_id_str = std::to_string(cam_id);
        double resize_factor =
                static_cast<double>(cameras_info[cam_id_str]["resize_factor"]);
        std::vector<double> cam_params(11, 0);
        cam_params[0] =
                static_cast<double>(cameras_info[cam_id_str]["width_px"]) /
                resize_factor;
        cam_params[1] =
                static_cast<double>(cameras_info[cam_id_str]["height_px"]) /
                resize_factor;

        if (cameras_info[cam_id_str]["intrinsics"].size()) {
            cam_params[2] = static_cast<double>(
                                    cameras_info[cam_id_str]["intrinsics"][0]) /
                            resize_factor;
            cam_params[3] = static_cast<double>(
                                    cameras_info[cam_id_str]["intrinsics"][1]) /
                            resize_factor;
            cam_params[4] = static_cast<double>(
                                    cameras_info[cam_id_str]["intrinsics"][2]) /
                            resize_factor;
            cam_params[5] = static_cast<double>(
                                    cameras_info[cam_id_str]["intrinsics"][3]) /
                            resize_factor;
        } else if (cameras_info[cam_id_str]["exif_intrinsics"].size()) {
            cam_params[2] =
                    static_cast<double>(
                            cameras_info[cam_id_str]["exif_intrinsics"][0]) /
                    resize_factor;
            cam_params[3] =
                    static_cast<double>(
                            cameras_info[cam_id_str]["exif_intrinsics"][1]) /
                    resize_factor;
            cam_params[4] =
                    static_cast<double>(
                            cameras_info[cam_id_str]["exif_intrinsics"][2]) /
                    resize_factor;
            cam_params[5] =
                    static_cast<double>(
                            cameras_info[cam_id_str]["exif_intrinsics"][3]) /
                    resize_factor;
        } else {
            utility::LogError("Camera parameters not found for camera id {}.",
                              cam_id_str);
        }

        if (cameras_info[cam_id_str]["distortion_params"].size()) {
            cam_params[6] = cameras_info[cam_id_str]["distortion_params"][0];
            cam_params[7] = cameras_info[cam_id_str]["distortion_params"][1];
            cam_params[8] = cameras_info[cam_id_str]["distortion_params"][2];
            cam_params[9] = cameras_info[cam_id_str]["distortion_params"][3];
            cam_params[10] = cameras_info[cam_id_str]["distortion_params"][4];
        }

        image_id_to_camera_params[image_id] = cam_params;
    }

    return image_id_to_camera_params;
}

int main(int argc, char* argv[]) {
    using namespace open3d;
    using json = nlohmann::json;

    // if (argc != 3 ||
    //     utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
    //     PrintHelp();
    //     return 1;
    // }

    std::string dataset_name =
            utility::GetProgramOptionAsString(argc, argv, "--dataset_name", "");
    std::string chunk_info_path =
            utility::GetProgramOptionAsString(argc, argv, "--chunk_info", "");
    std::string images_info_path =
            utility::GetProgramOptionAsString(argc, argv, "--images_info", "");
    std::string cameras_info_path =
            utility::GetProgramOptionAsString(argc, argv, "--cameras_info", "");

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // TODO: Add sanity checks.
    // 		 - Check if files exists.
    // 		 - Sanity checks on image_ids and num_images.
    std::ifstream chunk_info_stream(chunk_info_path);
    json chunk_info;
    chunk_info_stream >> chunk_info;

    const size_t num_images = chunk_info["matched_images"].size();

    // Get image_ids and path_to_keypoints.
    std::vector<int> image_ids;
    image_ids.reserve(num_images);
    std::unordered_map<int, std::string> path_to_keypoints_files;
    path_to_keypoints_files.reserve(num_images);
    std::string kpts_root_dir =
            static_cast<std::string>(chunk_info["kpts_folder"]);
    for (size_t i = 0; i < num_images; i++) {
        const int image_id = chunk_info["matched_images"][i];
        image_ids.push_back(image_id);
        const std::string path_to_kpt =
                kpts_root_dir + "kpts_" + std::to_string(image_id) + ".bin";
        path_to_keypoints_files[image_id] = path_to_kpt;
    }

    auto image_id_to_cam_params = GetCameraParamsFromInfoFiles(
            image_ids, images_info_path, cameras_info_path);

    // Get image match pairs.
    std::vector<std::pair<int, int>> match_pairs;
    const size_t num_pairs = chunk_info["image_pairs"].size() / 2;
    match_pairs.reserve(num_pairs);
    for (size_t i = 0; i < num_pairs; i++) {
        match_pairs.push_back(
                std::make_pair(chunk_info["image_pairs"][2 * i],
                               chunk_info["image_pairs"][2 * i + 1]));
    }

    const std::string path_to_output_eg_file = "/home/rey/eg_test.json";
    const std::string path_to_output_trackinfo_file =
            "/home/rey/trackinfo_test.json";

    open3d::preimage::image::MatchFeatures(
            image_ids, match_pairs, image_id_to_cam_params,
            path_to_keypoints_files, path_to_output_eg_file,
            path_to_output_trackinfo_file, 0.85, 0.95, 0.85, 0.95, false, 0);

    return 0;
}

/*
    image-io :
        - reads settings.json, dataset.json, and images from a dataset
        - writes detection_chunk_id.json [for id in range(0, num_chunks)]
        - writes image_chunk_id.json [for id in range(0, num_chunks)]
        - writes camera_chunk_id.json [for id in range(0, num_chunks)]
        - writes image_name_to_id_chunk_id.json [for id in range(0, num_chunks)]
        - writes med_images_chunk_id.npy [for id in range(0, num_chunks)]

    feature-detection : [for chunk_id in range(0, num_chunks)]
        - reads detection_chunk_id.json [reads image_id, keypoints_path from it]
        - dumps keypoints_path/kpts_id.bin [for id in image_id]
*/
