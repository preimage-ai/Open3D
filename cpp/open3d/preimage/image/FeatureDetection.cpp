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

#include "open3d/preimage/image/FeatureDetection.h"

#include "open3d/t/io/ImageIO.h"

namespace open3d {
namespace preimage {

FeatureDetection::FeatureDetection(int sample_val) : sample_(sample_val) {
    std::cout << " DEBUG #" << std::endl;
}

// FeatureDetection::~FeatureDetection() {}

// void FeatureDetection::DetectAndSaveFeatures(std::string image_path,
//                                              std::string output_dir_path) {
//     // Read image
//     auto image = t::io::CreateImageFromFile(image_path);
//     if (image == nullptr) {
//         utility::LogError("Failed to read image: {}", image_path);
//     }

//     // Detect features
//     // auto features = DetectFeatures(*image);
//     // if (features == nullptr) {
//     //     utility::LogError("Failed to detect features");
//     // }

//     // Save features
//     std::cout << "Saved features for " << image_path << std::endl;
//     // io::WriteFeature(output_dir_path + "/features.json", *features);
// }

}  // namespace preimage
}  // namespace open3d
