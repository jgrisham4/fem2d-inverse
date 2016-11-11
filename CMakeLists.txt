cmake_minimum_required(VERSION 3.6)
project(fem2d)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(include)
file(GLOB HEADER_FILES "include/*.h")
file(GLOB SOURCE_FILES "src/*.cpp")
add_subdirectory(src)
