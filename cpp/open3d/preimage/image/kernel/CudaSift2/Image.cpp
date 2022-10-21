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

// #include "open3d/core/Dtype.h"
// #include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/io/ImageIO.h"
#include <vips/vips.h>

namespace open3d {
namespace preimage {
namespace kernel {

FeatureDetector::FeatureDetector(const std::string &source_image_path,
                                 const std::string &output_feature_path)
    : source_image_path_(source_image_path),
      output_feature_path_(output_feature_path) {
    std::cout << "Computing SIFT features for " << source_image_path_
              << std::endl;
    InitCuda(0);
}

// static void rootsift(float (&desc)[128]) {
//     float sum = 0;
//     sum = std::accumulate(desc, desc + 128, sum);
//     for (size_t i = 0; i < 128; i++) {
//         desc[i] = desc[i] / sum;
//         desc[i] = sqrt(desc[i]);
//     }
// }

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
        core::Tensor source_image_tensor,
        int64_t w,
        int64_t h,
        const std::string output_feature_path) {
    void *gbuf = source_image_tensor.GetDataPtr();
    int array_sz = w * h;
    // Extract image as a buffer
    // Cast as unsigned char (8-bit)
    unsigned char *bw_img_arr = reinterpret_cast<unsigned char *>(gbuf);
    // Create float array (32-bit)
    float *fimg = (float *)malloc(int(array_sz) * sizeof(float));
    std::copy(bw_img_arr, bw_img_arr + int(array_sz), fimg);

    // Copy image to GPU, this function requires a float array
    CudaImage cudaImg;
    std::cout << "Starting allocation." << std::endl;
    int p = iAlignUp(w, 128);
    std::cout << "pitch set: " << p << std::endl;
    cudaImg.Allocate(w, h, p, false, NULL, fimg);
    std::cout << "pitch set: " << p << std::endl;

    std::cout << "Downloading to device." << std::endl;
    // cudaImg.DownloadFromTensor(source_image_tensor, core::Device("CUDA:0"));
    cudaImg.Download();
    // cudaImg.d_data =
    // source_image_tensor.To(core::Device("CUDA:0")).GetDataPtr<float>();
    // auto timg3 = core::Tensor(cudaImg.d_data, core::SizeVector({p * h}),
    //                           core::Dtype::Float32, core::Device("CUDA:0"))
    //                      .Reshape({h, p});
    // std::cout << "CUDA Image host tensor: " << timg3.GetShape().ToString()
    //           << std::endl;
    // timg3 = timg3.Mul(255.0).To(core::Dtype::UInt8);
    // auto image3 = t::geometry::Image(timg3).To(core::Device("CPU:0"));
    // t::io::WriteImage("/home/rey/data/tmp/debug2.jpg", image3);

    // std::cout << "Download Complete." << std::endl;

    // cudaImg.Readback();
    // std::cout << "Readback Complete." << std::endl;
    // auto timg4 = core::Tensor(cudaImg.h_data, core::SizeVector({w * h}),
    //                           core::Dtype::Float32, core::Device("CPU:0"))
    //                      .Reshape({h, w});
    // std::cout << "Passed tensor: " << timg4.GetShape().ToString() << std::endl;
    // timg4 = timg4.Mul(255.0).To(core::Dtype::UInt8);
    // auto image = t::geometry::Image(timg4);
    // t::io::WriteImage("/home/rey/data/tmp/debug3.jpg", image);

    siftMemoryTmp_ = AllocSiftTempMemory(w, h, 5, false);
    std::cout << "Assigned Temp Memory to Device for SIFT data." << std::endl;
    InitSiftData(siftData_, 32768, true, true);
    std::cout << "SiftData allocated" << std::endl;

#define VERBOSE
    ExtractSift(siftData_, cudaImg, 5, params_.initBlur, params_.thresh, 0.0f,
                false, siftMemoryTmp_);
    const unsigned int num_features = siftData_.numPts;
    std::cout << "Number of features extracted: " << num_features << std::endl;

    SaveFeaturesBinFile(output_feature_path);
    FreeSiftData(siftData_);
    return num_features;
}

}  // namespace kernel
}  // namespace preimage
}  // namespace open3d
