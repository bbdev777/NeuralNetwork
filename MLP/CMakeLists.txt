cmake_minimum_required(VERSION 3.10)
project(MLP_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES
    main.cpp
    mlp.cpp
)


set(HEADERS
    mlp.h
)


add_executable(MLP ${SOURCES} ${HEADERS})

