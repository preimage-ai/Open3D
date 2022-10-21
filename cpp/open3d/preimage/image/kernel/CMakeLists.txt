open3d_ispc_add_library(preimage_image_kernel OBJECT)

find_package(OpenCV REQUIRED)

if (BUILD_CUDA_MODULE)
    target_sources(preimage_image_kernel PRIVATE
        CudaSift/cudaImage.cu 
        CudaSift/cudaSiftD.cu
        CudaSift/cudaSiftH.cu
        CudaSift/matching.cu
    )
endif()

target_sources(preimage_image_kernel PRIVATE
        CudaSift/Image.cpp
)

open3d_show_and_abort_on_warning(preimage_image_kernel)
open3d_set_global_properties(preimage_image_kernel)
open3d_set_open3d_lib_properties(preimage_image_kernel)
open3d_link_3rdparty_libraries(preimage_image_kernel)

if(BUILD_CUDA_MODULE)
    target_include_directories(preimage_image_kernel SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
