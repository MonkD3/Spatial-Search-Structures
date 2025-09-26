
# ===================== geometry ===========================
target_include_directories(geometry
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
           $<INSTALL_INTERFACE:include>
)

# Set header when installing target 
target_sources(geometry PUBLIC
    FILE_SET geometry_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/bbox.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/bbox.h
          ${CMAKE_CURRENT_SOURCE_DIR}/point.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/point.h
          ${CMAKE_CURRENT_SOURCE_DIR}/predicates.hpp
)

# ==================== quadtree ===========================

target_include_directories(quadtree
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
           $<INSTALL_INTERFACE:include>
)

# Set public installed include
target_sources(quadtree PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES ${CMAKE_SOURCE_DIR}/include/quadtree.hpp
          ${CMAKE_SOURCE_DIR}/include/indexed_quadtree.hpp

)
