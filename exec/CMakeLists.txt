
add_executable(main)
target_sources(main PUBLIC main.cpp)
target_include_directories(main PUBLIC ../include)

add_executable(nbody)
target_sources(nbody PUBLIC nbody.cpp)
target_include_directories(nbody PUBLIC ../include)


add_executable(wn)
target_sources(wn PUBLIC fast_winding_number.cpp)
target_include_directories(wn PUBLIC ../include)

find_library(gmsh gmsh)
find_package(OpenMP)
add_executable(gmsh_wn)
target_sources(gmsh_wn PUBLIC fast_winding_number_gmsh.cpp)
target_include_directories(gmsh_wn PUBLIC ../include)
target_link_libraries(gmsh_wn PUBLIC ${gmsh} omp)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
