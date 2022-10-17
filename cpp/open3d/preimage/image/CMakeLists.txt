open3d_ispc_add_library(preimage_image OBJECT)

# add_subdirectory(kernel)

target_sources(preimage_image PRIVATE
    FeatureDetection.cpp
    Image.cpp
)

open3d_show_and_abort_on_warning(preimage_image)
open3d_set_global_properties(preimage_image)
open3d_set_open3d_lib_properties(preimage_image)
open3d_link_3rdparty_libraries(preimage_image)
