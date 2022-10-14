open3d_ispc_add_library(preimage_feature_detection_kernel OBJECT)

find_package(OpenCV REQUIRED)

if (BUILD_CUDA_MODULE)
    target_sources(preimage_feature_detection_kernel PRIVATE
        CudaSift/cudaImage.cu 
        CudaSift/cudaSiftH.cu  
        CudaSift/matching.cu
    )
endif()

# target_sources(preimage_feature_detection_kernel PRIVATE
#     CudaSift/geomFuncs.cpp  
#     CudaSift/mainSift.cpp  
# )

open3d_show_and_abort_on_warning(preimage_feature_detection_kernel)
open3d_set_global_properties(preimage_feature_detection_kernel)
open3d_set_open3d_lib_properties(preimage_feature_detection_kernel)
open3d_link_3rdparty_libraries(preimage_feature_detection_kernel)

if(BUILD_CUDA_MODULE)
    target_include_directories(preimage_feature_detection_kernel SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
