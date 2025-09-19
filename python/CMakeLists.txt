set(PYTHON_API_SRC
  __init__.py
  spatial_structures.py
)

foreach(f ${PYTHON_API_SRC})
    configure_file("${f}" "${PYTHON_BUILD_DIR}/${f}" @ONLY)
endforeach()

# The 'DESTINATION' should be a location that is included in the user's PYTHON_PATH.
install(
    DIRECTORY ${PYTHON_BUILD_DIR}/
    DESTINATION "${CMAKE_INSTALL_PREFIX}/search_structures"
    PATTERN "*.py"
)

