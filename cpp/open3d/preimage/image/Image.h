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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace preimage {
namespace image {

void RunFD();

///
/// \class Image
///
/// \brief Image class for storing image data.
/// \param tensor_images : Tensor of shape (NUM_IMAGES, HEIGHT, WIDTH) of
/// stacked images of type uint8_t, shape {HEIGHT, WIDTH} (grayscale).
/// \param output_feature_path : Path to output directory to dump npy files for
/// each image by image_filenames as indexed.
bool DetectAndSaveSIFTFeatures(const core::Tensor& tensor_images,
                               const std::vector<std::string>& image_filenames,
                               const float init_blur = 1.0,
                               const float thresh = 1.0,
                               const int octaves = 5,
                               const float min_scale = 0.25,
                               const bool upscale = false);

}  // namespace image
}  // namespace preimage
}  // namespace open3d
