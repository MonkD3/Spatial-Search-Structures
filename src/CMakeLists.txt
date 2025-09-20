add_library(geometry SHARED)
add_library(SearchStructures::geometry ALIAS geometry)

target_include_directories(geometry
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

# Set .cpp files
target_sources(geometry
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/point.cpp
)

# Set public headers
target_sources(geometry PUBLIC
    FILE_SET geometry_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES ${CMAKE_SOURCE_DIR}/include/bbox.hpp
          ${CMAKE_SOURCE_DIR}/include/bbox.h
          ${CMAKE_SOURCE_DIR}/include/point.hpp
          ${CMAKE_SOURCE_DIR}/include/point.h
          ${CMAKE_SOURCE_DIR}/include/predicates.hpp
)

if (TARGET geometry)
    install(
        TARGETS geometry
        EXPORT SearchStructuresTargets
        FILE_SET geometry_public_headers
    )
endif()


add_library(quadtree SHARED)
add_library(SearchStructures::quadtree ALIAS quadtree)
target_link_libraries(quadtree PRIVATE geometry)

target_include_directories(quadtree
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/interface>
           $<INSTALL_INTERFACE:include>
           $<INSTALL_INTERFACE:interface>
)

# Set .cpp files
target_sources(quadtree
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_2D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_3D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/surface_distance_2D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh_node_loc_2D.cpp
)

# Set public headers
target_sources(quadtree PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES ${CMAKE_SOURCE_DIR}/include/quadtree.hpp
)

# Set public interface
target_sources(quadtree PUBLIC
    FILE_SET interface_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/interface
    FILES ${CMAKE_SOURCE_DIR}/interface/winding_number_tree_2D.h
          ${CMAKE_SOURCE_DIR}/interface/winding_number_tree_3D.h
          ${CMAKE_SOURCE_DIR}/interface/surface_distance_2D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mesh_node_loc_2D.h
)


if (TARGET quadtree)
    install(
        TARGETS quadtree  
        EXPORT SearchStructuresTargets
        FILE_SET public_headers 
        FILE_SET interface_headers
    )
endif()

install(
    EXPORT SearchStructuresTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SearchStructures
    NAMESPACE SearchStructures::
    FILE SearchStructuresTargets.cmake # Not sure if this is still needed
)

include(CMakePackageConfigHelpers)
configure_package_config_file( 
  "Config.cmake.in" 
  "SearchStructuresConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SearchStructures
  PATH_VARS
    CMAKE_INSTALL_LIBDIR
  )

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SearchStructuresConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
  )

### Install Config and ConfigVersion files
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/SearchStructuresConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SearchStructuresConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SearchStructures"
  )

