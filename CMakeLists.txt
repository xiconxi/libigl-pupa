cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)

option(PMP_BUILD_APPS     "Build the PMP applications"  OFF)
option(PMP_BUILD_EXAMPLES "Build the PMP examples"      OFF)
option(PMP_BUILD_TESTS    "Build the PMP test programs" OFF)
option(PMP_BUILD_DOCS     "Build the PMP documentation" OFF)
option(PMP_BUILD_VIS      "Build the PMP visualization tools" OFF)
option(PMP_INSTALL        "Install the PMP library and headers" OFF)

#set(PMP_BUILD_APPS false CACHE BOOL "Build the PMP examples")
#set(PMP_BUILD_EXAMPLES false CACHE BOOL "Build the PMP examples")
#set(PMP_BUILD_TESTS false CACHE BOOL "Build the PMP test programs")
#set(PMP_BUILD_DOCS false CACHE BOOL "Build the PMP documentation")
#set(PMP_BUILD_VIS false CACHE BOOL "Build the PMP visualization tools")
#set(PMP_INSTALL false CACHE BOOL "Install the PMP library and headers")

include_directories(./external/pmp-library/src)

find_package(LIBIGL REQUIRED QUIET)
add_subdirectory(external/pmp-library)
add_subdirectory(src/03_Smoothing)
