cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(SearchStructures CXX)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(PYTHON_BUILD_DIR "${PROJECT_BINARY_DIR}/python")

set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

endif()
option(ENABLE_EXAMPLES "Compile the C++ examples" OFF)

add_compile_options(-Wall -Werror -Wshadow -Wextra -march=native)

find_library(gmsh gmsh)
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ======================= Define the targets ======================

add_library(geometry SHARED)
add_library(SearchStructures::geometry ALIAS geometry)

add_library(quadtree SHARED)
add_library(SearchStructures::quadtree ALIAS quadtree)

# ======================== Add directories =======================
add_subdirectory(interface/)  # library interface
add_subdirectory(include/)    # Include files
add_subdirectory(python/)     # Python bindings
add_subdirectory(src/)        # Source files

if (ENABLE_EXAMPLES)
    add_subdirectory(examples/)
endif()

