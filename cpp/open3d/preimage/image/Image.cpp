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

#include "open3d/t/geometry/Image.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/preimage/image/kernel/CudaSift/Image.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace preimage {

void RunFD() {
    std::string input_image_path =
            "/home/rey/data/tmp/EP-11-16323_0011_0468.JPG";
    std::string output_feature_path = "/home/rey/data/tmp/0.bin";

    t::geometry::Image image;
    core::Tensor image_tensor;
    t::io::ReadImage(input_image_path, image);
    image = image.Resize(0.5);
    int64_t w = image.GetCols();
    int64_t h = image.GetRows();

    if (image.GetDtype() == core::Dtype::UInt8) {
        utility::LogInfo("Image Dtype: {}, Shape: {}",
                         image.AsTensor().GetDtype().ToString(),
                         image.AsTensor().GetShape().ToString());

        image_tensor = image.AsTensor()
                               .To(core::Float32)
                               .Mean({2}, false)
                               .Flatten()
                               .To(core::UInt8)
                               .Contiguous();

        utility::LogInfo("Image Tensor Dtype: {}, Shape: {}",
                         image_tensor.GetDtype().ToString(),
                         image_tensor.GetShape().ToString());
    } else {
        utility::LogError("Image Dtype: {} not supported",
                          image.GetDtype().ToString());
    }

    open3d::preimage::kernel::FeatureDetector fd(input_image_path,
                                                 output_feature_path);
    unsigned int num_features =
            fd.DetectAndSaveFeatures(image_tensor, w, h, output_feature_path);
    std::cout << "Number of features: " << num_features << std::endl;
}

}  // namespace preimage
}  // namespace open3d
