cmake_minimum_required(VERSION 3.16)
project(inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gpt2 main.cpp)
target_include_directories(gpt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gpt2 PRIVATE -O3 -march=native -ffast-math -Wall -Wextra)

find_package(Threads REQUIRED)
target_link_libraries(gpt2 PRIVATE Threads::Threads)
