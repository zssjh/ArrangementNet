cmake_minimum_required(VERSION 3.9.0)

project(MeshRecon)

set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(ENABLE_COVERAGE "Use gcov" OFF)
option(FIND_PACKAGE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -mavx -fPIC -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -fPIC -pthread")
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_BUILD_TYPE "debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_library(
    graphcut SHARED
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gco/block.h
    ${CMAKE_CURRENT_LIST_DIR}/gco/energy.h
    ${CMAKE_CURRENT_LIST_DIR}/gco/GCoptimization.h
    ${CMAKE_CURRENT_LIST_DIR}/gco/GCoptimization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gco/graph.h
    ${CMAKE_CURRENT_LIST_DIR}/gco/graph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gco/LinkedBlockList.h
    ${CMAKE_CURRENT_LIST_DIR}/gco/LinkedBlockList.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gco/maxflow.cpp
)
target_link_libraries(
    graphcut
)
