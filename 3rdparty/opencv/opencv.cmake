include(ExternalProject)

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_opencv
    PREFIX opencv
    URL https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz
    URL_HASH SHA256=1ec1cba65f9f20fe5a41fda1586e01c70ea0c9a6d7b67c9e13edf0cfe2239277
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/opencv"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_BUILD_EXAMPLES=OFF
        -DBUILD_TESTS=OFF
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}opencv_core${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_opencv INSTALL_DIR)
set(OPENCV_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(OPENCV_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(OPENCV_LIBRARIES opencv)
