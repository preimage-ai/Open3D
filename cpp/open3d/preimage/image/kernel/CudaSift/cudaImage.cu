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

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>

#include "cudaImage.h"
#include "cudautils.h"

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

void CudaImage::Allocate(
        int w, int h, int p, bool host, float *devmem, float *hostmem) {
    width = w;
    height = h;
    pitch = p;
    d_data = devmem;
    h_data = hostmem;
    t_data = NULL;
    if (devmem == NULL) {
        safeCall(cudaMallocPitch((void **)&d_data, (size_t *)&pitch,
                                 (size_t)(sizeof(float) * width),
                                 (size_t)height));
        pitch /= sizeof(float);
        if (d_data == NULL) printf("Failed to allocate device data\n");
        d_internalAlloc = true;
    }
    if (host && hostmem == NULL) {
        h_data = (float *)malloc(sizeof(float) * pitch * height);
        h_internalAlloc = true;
    }
}

CudaImage::CudaImage()
    : width(0),
      height(0),
      h_data(NULL),
      d_data(NULL),
      t_data(NULL),
      d_internalAlloc(false),
      h_internalAlloc(false) {}

CudaImage::~CudaImage() {
    if (d_internalAlloc && d_data != NULL) safeCall(cudaFree(d_data));
    d_data = NULL;
    if (h_internalAlloc && h_data != NULL) free(h_data);
    h_data = NULL;
    if (t_data != NULL) safeCall(cudaFreeArray((cudaArray *)t_data));
    t_data = NULL;
}

double CudaImage::Download() {
    TimerGPU timer(0);
    int p = sizeof(float) * pitch;
    if (d_data != NULL && h_data != NULL)
        safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width,
                              sizeof(float) * width, height,
                              cudaMemcpyHostToDevice));
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("Download time =               %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

double CudaImage::Readback() {
    TimerGPU timer(0);
    int p = sizeof(float) * pitch;
    safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p,
                          sizeof(float) * width, height,
                          cudaMemcpyDeviceToHost));
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("Readback time =               %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

double CudaImage::InitTexture() {
    TimerGPU timer(0);
    cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
    safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
    if (t_data == NULL) printf("Failed to allocated texture data\n");
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

double CudaImage::CopyToTexture(CudaImage &dst, bool host) {
    if (dst.t_data == NULL) {
        printf("Error CopyToTexture: No texture data\n");
        return 0.0;
    }
    if ((!host || h_data == NULL) && (host || d_data == NULL)) {
        printf("Error CopyToTexture: No source data\n");
        return 0.0;
    }
    TimerGPU timer(0);
    if (host)
        safeCall(cudaMemcpy2DToArray(
                (cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float) * pitch,
                sizeof(float) * pitch, dst.height, cudaMemcpyHostToDevice));
    else
        safeCall(cudaMemcpy2DToArray(
                (cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float) * pitch,
                sizeof(float) * pitch, dst.height, cudaMemcpyDeviceToDevice));
    safeCall(cudaDeviceSynchronize());
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}
