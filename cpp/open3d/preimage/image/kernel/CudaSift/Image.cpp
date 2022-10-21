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

#include "open3d/preimage/image/kernel/CudaSift/Image.h"

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "open3d/preimage/image/kernel/CudaSift/cudaSift.h"

// #include "open3d/core/Dtype.h"
// #include "open3d/core/Tensor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Logging.h"
// #include <vips/vips.h>

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

FeatureDetector::FeatureDetector(
        const core::Tensor &images_tensor,
        const std::vector<std::string> &output_filenames,
        const float init_blur,
        const float thresh,
        const int octaves,
        const float min_scale,
        const bool upscale,
        const int max_features)
    : images_tensor_(images_tensor),
      output_filenames_(output_filenames),
      init_blur_(init_blur),
      thresh_(thresh),
      octaves_(octaves),
      min_scale_(min_scale),
      upscale_(upscale) {
    const open3d::core::SizeVector images_tensor_shape =
            images_tensor.GetShape();
    num_images_ = images_tensor_shape[0];
    height_ = images_tensor_shape[1];
    width_ = images_tensor_shape[2];
    pitch_ = iAlignUp(width_, 128);

    if (!images_tensor_.GetDevice().IsCUDA()) {
        utility::LogError("Feature Detection only supported on CUDA devices.");
    }
    // if (!images_tensor_.GetDtype() != core::Dtype::Float32) {
    //     utility::LogError(
    //             "Feature Detection only supports Float32 type
    //             images_tensor.");
    // }

    InitCuda(0);
    InitSiftData(siftData_, max_features, true, true);
    if (upscale) {
        siftMemoryTmp_ =
                AllocSiftTempMemory(2 * width_, 2 * height_, octaves, false);
    } else {
        siftMemoryTmp_ = AllocSiftTempMemory(width_, height_, octaves, false);
    }

    for (int idx = 0; idx < num_images_; ++idx) {
        auto tensor = images_tensor[idx].Flatten();
        float *img = tensor.GetDataPtr<float_t>();
        uint32_t num_features = DetectAndSaveFeatures(img, idx);
        utility::LogDebug(" {} keypoints saved at {}.", num_features,
                          output_filenames[idx]);
    }
}

FeatureDetector::~FeatureDetector() { FreeSiftData(siftData_); }

void FeatureDetector::SaveFeaturesBinFile(
        const std::string output_feature_path) {
    // xval, yval, scale, orientation, 128 floats in descriptors = 144 bytes
    // total
    float *siftDataTmp =
            (float *)malloc(int(siftData_.numPts) * sizeof(float) * 144);
    for (int i = 0; i < siftData_.numPts; i++) {
        siftDataTmp[i * 144] = siftData_.h_data[i].xpos;
        siftDataTmp[i * 144 + 1] = siftData_.h_data[i].ypos;
        siftDataTmp[i * 144 + 2] = siftData_.h_data[i].scale;
        siftDataTmp[i * 144 + 3] = siftData_.h_data[i].sharpness;
        siftDataTmp[i * 144 + 4] = siftData_.h_data[i].edgeness;
        siftDataTmp[i * 144 + 5] = siftData_.h_data[i].orientation;
        siftDataTmp[i * 144 + 6] = siftData_.h_data[i].score;
        siftDataTmp[i * 144 + 7] = siftData_.h_data[i].ambiguity;
        siftDataTmp[i * 144 + 8] = float(siftData_.h_data[i].match);
        siftDataTmp[i * 144 + 9] = siftData_.h_data[i].match_xpos;
        siftDataTmp[i * 144 + 10] = siftData_.h_data[i].match_ypos;
        siftDataTmp[i * 144 + 11] = siftData_.h_data[i].match_error;
        siftDataTmp[i * 144 + 12] = siftData_.h_data[i].subsampling;
        siftDataTmp[i * 144 + 13] = siftData_.h_data[i].empty[0];
        siftDataTmp[i * 144 + 14] = siftData_.h_data[i].empty[1];
        siftDataTmp[i * 144 + 15] = siftData_.h_data[i].empty[2];
        // rootsift(siftData_.h_data[i].data);
        std::copy(siftData_.h_data[i].data, siftData_.h_data[i].data + 128,
                  siftDataTmp + i * 144 + 16);
    }
    std::stringstream ss;
    ss << output_feature_path;
    FILE *file = fopen(ss.str().c_str(), "wb");
    fwrite(siftDataTmp, sizeof(float), int(siftData_.numPts) * 144, file);
    free(siftDataTmp);
    fclose(file);
}

unsigned int FeatureDetector::DetectAndSaveFeatures(float *data_ptr,
                                                    const int image_id) {
    CudaImage cudaImg;
    // TODO: Add pitch padding support in tensor manually to avoid
    // re-allocations to cudaImg.
    // TODO: Remove the need for cudaImg and use tensor directly.
    cudaImg.Allocate(width_, height_, pitch_, false, data_ptr, NULL);
    ExtractSift(siftData_, cudaImg, octaves_, init_blur_, thresh_, min_scale_,
                upscale_, siftMemoryTmp_);
    // SaveFeaturesBinFile(image_id, output_feature_path_);
    int num_features = siftData_.numPts;
    return num_features;
}

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
