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

int main(int argc, char *argv[]) {
    using namespace open3d;
    using json = nlohmann::json;

    // if (argc != 3 ||
    //     utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
    //     PrintHelp();
    //     return 1;
    // }

    std::string dataset_name =
            utility::GetProgramOptionAsString(argc, argv, "--dataset_name", "");
    std::string chunk_json_path = utility::GetProgramOptionAsString(
            argc, argv, "--chunk_json_path", "");
    std::string device =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    std::ifstream datasrc(chunk_json_path);
    json j;
    datasrc >> j;
    std::cout << "Found " << j["image_ids"].size()
              << " images for current chunk. " << std::endl;

    open3d::core::Tensor images_tensor =
            open3d::core::Tensor::Load(j["medimgs_path"]);
    std::cout << "Shape of medimgs: " << images_tensor.GetShape().ToString()
              << std::endl;

    std::vector<std::string> out_filename;
    for (auto it : j["image_ids"]) {
        std::string out_string = std::string(j["keypoints_path"]) + "/kpts_" +
                                 std::to_string(static_cast<int>(it)) + ".bin";
        out_filename.push_back(out_string);
    }
    const float init_blur = static_cast<float>(j["sift_init_blur"]);
    const float thresh = static_cast<float>(j["sift_thres"]);
    const int octaves = static_cast<int>(j["sift_octaves"]);
    const float min_scale = static_cast<float>(j["sift_min_scale"]);
    const bool upscale = static_cast<bool>(j["sift_upscale"]);
    const int max_num_keypoints = static_cast<int>(j["sift_max_keypoints"]);

    // Handle this float32, grayscale conversion in previous node.
    auto tensor = images_tensor.To(core::Dtype::Float32)
                          .Mean({3}, false)
                          .To(core::Device("CUDA:0"), true);

    open3d::preimage::image::DetectAndSaveSIFTFeatures(
            tensor, out_filename, init_blur, thresh, octaves, min_scale,
            upscale, max_num_keypoints);
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
