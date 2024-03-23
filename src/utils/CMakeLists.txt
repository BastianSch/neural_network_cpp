set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/utils/Matrix.h")

add_library(matrix SHARED Matrix.cu ${HEADER_LIST})

target_include_directories(matrix PUBLIC ../../include)

target_compile_features(matrix PUBLIC cxx_std_11)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})
