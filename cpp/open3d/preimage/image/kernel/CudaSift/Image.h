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

#pragma once

#include <string>

#include "open3d/core/Tensor.h"
#include "open3d/preimage/image/kernel/CudaSift/cudaSift.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

class FeatureDetector {
public:
    FeatureDetector(const core::Tensor& images_tensor_,
                    const std::vector<std::string>& output_filenames,
                    const float init_blur = 1.0,
                    const float thresh = 1.0,
                    const int octaves = 5,
                    const float min_scale = 0.0,
                    const bool upscale = false,
                    const int max_features = 32768);

    ~FeatureDetector();

    unsigned int DetectAndSaveFeatures(float* data_ptr, const int image_id);

    void SaveFeaturesBinFile(const std::string output_feature_path);

private:
    core::Tensor images_tensor_;
    std::vector<std::string> output_filenames_;

    float init_blur_;
    float thresh_;
    int octaves_;
    float min_scale_;
    bool upscale_;

    int num_images_ = 0;
    int height_ = 0;
    int width_ = 0;
    int pitch_ = 0;

    SiftData siftData_;
    float* siftMemoryTmp_;
};

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
