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
#include "open3d/t/geometry/Image.h"
#include "open3d/t/io/ImageIO.h"
// #include <vips/vips.h>

namespace open3d {
namespace preimage {
namespace image {
namespace kernel {

FeatureDetector::FeatureDetector(const std::string &source_image_path,
                                 const std::string &output_feature_path)
    : source_image_path_(source_image_path),
      output_feature_path_(output_feature_path) {
    std::cout << "Computing SIFT features for " << source_image_path_
              << std::endl;
    InitCuda(0);
    uint num_features;
    DetectAndSaveFeatures(source_image_path, 0, num_features);
    std::cout << "Number of features detected: " << num_features << std::endl;
    FreeSiftTempMemory(siftMemoryTmp_);
}

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

unsigned int FeatureDetector::DetectAndSaveFeatures(
        std::string filename,
        const unsigned int image_id,
        unsigned int &num_features) {
    t::geometry::Image image;
    t::io::ReadImage(filename, image);
    int w_ = image.GetCols();
    int h_ = image.GetRows();

    auto tensor = image.AsTensor()
                          .To(core::Dtype::Float32)
                          .Mean({2}, false)
                          .Flatten()
                          .To(core::Device("CUDA:0"), true);
    std::cout << "tensor shape: " << tensor.GetShape().ToString() << std::endl;
    float *dimg = tensor.GetDataPtr<float_t>();
    CudaImage cudaImg;
    cudaImg.Allocate(w_, h_, iAlignUp(w_, 128), false, dimg, NULL);
    siftMemoryTmp_ = AllocSiftTempMemory(w_, h_, 5, false);
    InitSiftData(siftData_, 32768, true, true);
    std::cout << "extracting... " << std::endl;
    ExtractSift(siftData_, cudaImg, 5, params_.initBlur, params_.thresh, 0.0f,
                false, siftMemoryTmp_);
    // saveBinFile(image_id, output_feature_path_);
    num_features = siftData_.numPts;
    std::cout << "extracted: " << num_features << std::endl;
    
    FreeSiftData(siftData_);
    // firstDetection_ = false;
    return num_features;
}

}  // namespace kernel
}  // namespace image
}  // namespace preimage
}  // namespace open3d
