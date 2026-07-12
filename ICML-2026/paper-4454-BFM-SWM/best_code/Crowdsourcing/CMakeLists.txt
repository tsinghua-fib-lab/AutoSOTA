cmake_minimum_required(VERSION 3.5)
project(CS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(CS main.cpp read_data.h SIP.h TripleEagleNm.h BFM-VM.h)