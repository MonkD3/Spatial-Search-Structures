add_library(quadtree SHARED)

target_include_directories(quadtree
    PUBLIC ${CMAKE_SOURCE_DIR}/include
           ${CMAKE_SOURCE_DIR}/interface
)

target_sources(quadtree
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/point.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_2D.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_3D.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/surface_distance_2D.cpp
    INTERFACE ${CMAKE_SOURCE_DIR}/include/bbox.hpp
              ${CMAKE_SOURCE_DIR}/include/bbox.h
              ${CMAKE_SOURCE_DIR}/include/point.hpp
              ${CMAKE_SOURCE_DIR}/include/point.h
              ${CMAKE_SOURCE_DIR}/interface/winding_number_tree_2D.h
              ${CMAKE_SOURCE_DIR}/interface/winding_number_tree_3D.h
              ${CMAKE_SOURCE_DIR}/interface/surface_distance2D.h
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

