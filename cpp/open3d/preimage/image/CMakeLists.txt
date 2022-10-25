add_subdirectory(kernel)

open3d_ispc_add_library(preimage_image OBJECT)

find_package (OpenCV REQUIRED)
include_directories ("/usr/include/opencv4/")

target_sources(preimage_image PRIVATE
    Image.cpp
)

open3d_show_and_abort_on_warning(preimage_image)
open3d_set_global_properties(preimage_image)
open3d_set_open3d_lib_properties(preimage_image)
open3d_link_3rdparty_libraries(preimage_image)

if(BUILD_CUDA_MODULE)
    target_include_directories(preimage_image SYSTEM PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
