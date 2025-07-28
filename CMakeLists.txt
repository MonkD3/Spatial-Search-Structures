cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(quadtree CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Werror -Wshadow -Wextra -march=native)
add_subdirectory(src/)


add_subdirectory(exec/)
