# ==================== quadtree ===========================

# Set public installed interface
target_sources(quadtree PUBLIC
    FILE_SET interface_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_2D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/winding_number_tree_3D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/surface_distance_2D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/indexed_mesh_node_loc_2D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mesh_node_loc_2D.h
          ${CMAKE_CURRENT_SOURCE_DIR}/point_cloud_nearest_neighbor.h
)

target_include_directories(quadtree
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
           $<INSTALL_INTERFACE:interface>
)
