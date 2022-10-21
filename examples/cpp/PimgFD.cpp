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

core::Tensor ReadImagesToTensor(std::vector<std::string> filenames) {
    core::Tensor images_tensor;
    for (auto filename : filenames) {
        VipsImage *vimg = vips_image_new_from_file(filename.c_str(), NULL);
        int h = vips_image_get_height(vimg);
        int w = vips_image_get_width(vimg);
        VipsImage *vimg_bw_;
        vips_colourspace(vimg, &vimg_bw_, VIPS_INTERPRETATION_B_W, NULL);
        size_t array_sz;
        // Extract image as a buffer
        uint8_t *gbuf_ = reinterpret_cast<uint8_t *>(
                vips_image_write_to_memory(vimg_bw_, &array_sz));
        // Convert buffer to a tensor
        core::Tensor img_tensor =
                core::Tensor(gbuf_, {h, w, 1}, core::Dtype::UInt8);
    }
    return images_tensor;
}

void PrintHelp() {
    using namespace open3d;

    // Currently reading images from path stored in
    // dataset_detectionn_chunk_X.json using VIPS. Later, directly read from
    // medimgs.npy.

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > PimgFD --dataset_name dataset --chunk_path dataset_detection_chunk_X.json --device CUDA:0");
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
    std::string chunk_path =
            utility::GetProgramOptionAsString(argc, argv, "--chunk_path", "");
    // TODO: Mode output_dir to JSON as working_dir
    std::string kpts_output_prefix = utility::GetProgramOptionAsString(
            argc, argv, "--kpts_output_prefix", "/home/rey/tmp/kpts_");
    // std::string images_tensor_path = utility::GetProgramOptionAsString(
    //         argc, argv, "--images_tensor_path", "");
    std::string device =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    std::ifstream datasrc(chunk_path);
    json j;
    datasrc >> j;
    std::cout << "Found " << j["image_ids"].size()
              << " images for current chunk. " << std::endl;

    open3d::core::Tensor images_tensor =
            open3d::core::Tensor::Load(j["med_imgs_location"]);
    std::cout << "Shape of medimgs: " << images_tensor.GetShape().ToString()
              << std::endl;

    std::vector<std::string> out_filename;
    for (auto it : j["image_ids"]) {
        std::string out_string = (std::string)kpts_output_prefix +
                                 std::to_string(static_cast<int>(it)) + ".bin";
        out_filename.push_back(out_string);
    }

    auto tensor = images_tensor.To(core::Dtype::Float32)
                          .Mean({3}, false)
                          .To(core::Device("CUDA:0"), true);

    open3d::preimage::image::DetectAndSaveSIFTFeatures(tensor, out_filename);
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
        - reads med_images_chunk_id.npy
        - reads detection_chunk_id.json [reads image_id to image_name from it,
   keypoint_path]
        - dumps keypoints_image_id.npy [for image_id in range(0, num_images)]
*/