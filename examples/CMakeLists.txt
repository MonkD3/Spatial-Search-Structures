set(CPP_EXEC
    main
    main3D
    fast_winding_number
    fast_winding_number_gmsh
)

foreach(f ${CPP_EXEC})
    add_executable(${f})
    target_sources(${f} PUBLIC "${f}.cpp")
    target_include_directories(${f} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${f} PUBLIC ${gmsh})
endforeach()

set(PYTHON_EXEC
    test_SurfaceTree2D.py
    test_WNOcTree.py
    test_WNQuadTree.py
    test_MeshTree2D.py
)

foreach(f ${PYTHON_EXEC})
    configure_file("${f}" "${PROJECT_BINARY_DIR}/examples/${f}" COPYONLY)
endforeach()

