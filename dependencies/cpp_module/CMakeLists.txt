cmake_minimum_required(VERSION 2.8.12)
project(flame_project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w -shared -std=c++11")

add_subdirectory(pybind11)
pybind11_add_module(get_vis module.cpp)
INCLUDE_DIRECTORIES("./eigen-eigen-323c052e1731")




