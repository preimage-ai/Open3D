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

#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudaImage.h"
#include "cudautils.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

int ExtractSiftLoop(SiftData &siftData,
                    CudaImage &img,
                    int numOctaves,
                    double initBlur,
                    float thresh,
                    float lowestScale,
                    float subsampling,
                    float *memoryTmp,
                    float *memorySub);
void ExtractSiftOctave(SiftData &siftData,
                       CudaImage &img,
                       int octave,
                       float thresh,
                       float lowestScale,
                       float subsampling,
                       float *memoryTmp);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj,
                           CudaImage &src,
                           SiftData &siftData,
                           int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj,
                              SiftData &siftData,
                              float subsampling,
                              int octave);
double OrientAndExtract(cudaTextureObject_t texObj,
                        SiftData &siftData,
                        float subsampling,
                        int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj,
                    CudaImage &baseImage,
                    CudaImage *results,
                    int octave);
double FindPointsMulti(CudaImage *sources,
                       SiftData &siftData,
                       float thresh,
                       float edgeLimit,
                       float factor,
                       float lowestScale,
                       float subsampling,
                       int octave);

#endif
