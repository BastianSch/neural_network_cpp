cmake_minimum_required(VERSION 3.14...3.28)

project(nn_cuda
	VERSION 0.0.1
        DESCRIPTION "Simple Neural Network with CUDA"
        LANGUAGES CXX CUDA
	)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  set(CMAKE_CXX_EXTENSIONS OFF)

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)

  find_package(Doxygen)

  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()

add_subdirectory(src)
add_subdirectory(apps)


