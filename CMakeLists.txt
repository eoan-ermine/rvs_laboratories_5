cmake_minimum_required(VERSION 3.28)
project(rvs_laboratories_5 LANGUAGES CXX CUDA)

add_executable(dataset_generator dataset_generator/dataset_generator.cpp)
add_executable(rvs_laboratories_5 rvs_laboratories_5/rvs_laboratories_5.cu)
