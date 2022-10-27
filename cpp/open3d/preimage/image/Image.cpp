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
#include <opencv4/opencv2/core.hpp>
#include <string>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/preimage/image/kernel/CudaSift/FeatureDetector.h"
#include "open3d/preimage/image/kernel/CudaSift/FeatureMatcher.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace preimage {
namespace image {

bool DetectAndSaveSIFTFeatures(const core::Tensor& tensor_images,
                               const std::vector<std::string>& output_filenames,
                               const float init_blur,
                               const float thresh,
                               const int octaves,
                               const float min_scale,
                               const bool upscale,
                               const int max_keypoints) {
    utility::Timer timer;
    timer.Start();
    open3d::preimage::image::kernel::FeatureDetector fd(
            tensor_images, output_filenames, init_blur, thresh, octaves,
            min_scale, upscale, max_keypoints);
    timer.Stop();
    utility::LogInfo("FeatureDetector took {} ms",
                     timer.GetDurationInMillisecond());
    // TODO: Add checks to kernel, and return false if something goes wrong.
    return true;
}

bool MatchFeatures(
        const std::vector<int>& image_ids,
        const std::vector<std::pair<int, int>>& match_pairs,
        const std::unordered_map<int, std::vector<double>>&
                image_id_to_cam_params,
        const std::unordered_map<int, std::string>& path_to_keypoints_files,
        const std::string& path_to_output_eg_file,
        const std::string& path_to_output_trackinfo_file,
        const double desc_thres,
        const double max_ratio,
        const double desc_thres_guided,
        const double max_ratio_guided,
        const bool undistort,
        const int device_id) {
    utility::Timer timer;
    timer.Start();
    open3d::preimage::image::kernel::FeatureMatcher fd(
            image_ids, match_pairs, image_id_to_cam_params,
            path_to_keypoints_files, path_to_output_eg_file,
            path_to_output_trackinfo_file, desc_thres, max_ratio,
            desc_thres_guided, max_ratio_guided, undistort, device_id);
    fd.Run();

    timer.Stop();
    utility::LogInfo("FeatureMatcher took {} ms",
                     timer.GetDurationInMillisecond());
    // TODO: Add checks to kernel, and return false if something goes wrong.
    return true;
}
/*
idx_to_image_id : list of indices of images, size: num_images
path_to_medimgs : path to medimgs
        medimgs : shape (num_images, height, width) of type float32, range
[0,1], grayscale. path_to_keypoints[image_id] : path to keypoints for image_id
        siftData_[idx]
*/

// bool MatchKeypointsBetweenImagePairs(
//         const std::vector<std::pair<int, int>>& images_pairs,
//         const std::vector<std::string>& image_filenames,
//         const std::vector<std::string>& output_filenames,
//         const float match_thresh,
//         const float match_ratio) {
//     utility::Timer timer;
//     timer.Start();
//     open3d::preimage::image::kernel::FeatureMatcher fm(
//             images_pairs, image_filenames, output_filenames, match_thresh,
//             match_ratio, max_keypoints);
//     timer.Stop();
//     utility::LogInfo("FeatureMatcher took {} ms",
//                      timer.GetDurationInMillisecond());
//     return true;
// }

}  // namespace image
}  // namespace preimage
}  // namespace open3d
