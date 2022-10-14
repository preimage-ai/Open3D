open3d_ispc_add_library(preimage_io OBJECT)

target_sources(preimage_io PRIVATE
    ImageIO.cpp
)

open3d_show_and_abort_on_warning(preimage_io)
open3d_set_global_properties(preimage_io)
open3d_set_open3d_lib_properties(preimage_io)
open3d_link_3rdparty_libraries(preimage_io)
