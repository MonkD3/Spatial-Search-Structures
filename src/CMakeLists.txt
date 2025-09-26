# Set .cpp files
target_sources(geometry
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/point.cpp
)

if (TARGET geometry)
    install(
        TARGETS geometry
        EXPORT SearchStructuresTargets
        FILE_SET geometry_public_headers
    )
endif()

target_link_libraries(quadtree PRIVATE geometry)

# Set .cpp files
target_sources(quadtree
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_2D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_3D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/surface_distance_2D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh_node_loc_2D.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/indexed_mesh_node_loc_2D.cpp

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

