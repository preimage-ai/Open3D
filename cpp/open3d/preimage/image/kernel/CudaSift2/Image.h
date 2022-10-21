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

// #include <limits>
// #include <memory>
#include <string>
// #include <vector>
// #include <iostream>

// #include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/preimage/image/kernel/CudaSift/cudaSift.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace preimage {
namespace kernel {

struct FeatureDetectorParams {
    float initBlur = 1.0;
    float thresh = 1.0;
    int octaves = 5;
    float minScale = 0.25;
    bool upscale = false;
};

class FeatureDetector {
public:
    FeatureDetector(const std::string& source_image_path,
                    const std::string& output_feature_path = "out.bin");

    virtual ~FeatureDetector() {}

    unsigned int DetectAndSaveFeatures(core::Tensor source_image_tensor,
                                       int64_t w,
                                       int64_t h,
                                       const std::string output_feature_path);

    void SaveFeaturesBinFile(const std::string output_feature_path);

private:
    FeatureDetectorParams params_;
    SiftData siftData_;
    float* siftMemoryTmp_;
    std::string source_image_path_;
    std::string output_feature_path_;
};

void Foo();

}  // namespace kernel
}  // namespace preimage
}  // namespace open3d
