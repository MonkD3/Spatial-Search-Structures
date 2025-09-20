set(PYTHON_API_SRC
  __init__.py
  fwn_2D.py
  fwn_3D.py
  surface_distance_2D.py
)

foreach(f ${PYTHON_API_SRC})
    configure_file("${f}" "${PYTHON_BUILD_DIR}/${f}" @ONLY)
endforeach()

# The 'DESTINATION' should be a location that is included in the user's PYTHON_PATH.
install(
    DIRECTORY ${PYTHON_BUILD_DIR}/
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/search_structures"
    PATTERN "*.py"
)

