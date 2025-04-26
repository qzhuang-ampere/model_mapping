cmake_minimum_required(VERSION 3.18)

# Set the project name
project(model_mapping)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Block to add Kestrel TR library
set(KESTREL_TR_DIR ${CMAKE_SOURCE_DIR}/externals/kestrel_tr)
set(KESTREL_TR_LIB kestrel_tr_lib)
set(KESTREL_TR_COMPILE_FLAGS Debug)
set(KESTREL_TR_LINK_FLAGS -Wl,--allow-multiple-definition)

# Find all .c files in KESTREL_TR_DIR
file(GLOB KESTREL_TR_SOURCES ${KESTREL_TR_DIR}/*.c ${KESTREL_TR_DIR}/*.cpp)

# Add a library target for the .c & .cpp files
add_library(${KESTREL_TR_LIB} ${KESTREL_TR_SOURCES})
target_compile_definitions(${KESTREL_TR_LIB} PRIVATE ${KESTREL_TR_COMPILE_FLAGS})
target_link_options(${KESTREL_TR_LIB} PRIVATE ${KESTREL_TR_LINK_FLAGS})

include_directories(${KESTREL_TR_DIR})

# Treat all warnings as errors
add_compile_options(-Wall -Werror -Wno-unknown-pragmas)

add_subdirectory(examples)
