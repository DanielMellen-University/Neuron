cmake_minimum_required(VERSION 3.20)

project(neuron
  VERSION 0.1.0
  DESCRIPTION "From-scratch C++ LLM inference engine for x86 CPUs"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEURON_AVX2 "Compile with AVX2 and FMA" OFF)
option(NEURON_AVX512 "Compile with AVX-512 (F, BW, DQ, VL, VNNI); implies AVX2" OFF)
option(NEURON_NATIVE "Compile with -march=native (overrides NEURON_AVX2/NEURON_AVX512)" OFF)
option(NEURON_WERROR "Treat warnings as errors" OFF)
option(NEURON_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NEURON_BUILD_TESTS "Build the unit tests" ON)

add_library(neuron_options INTERFACE)
target_compile_options(neuron_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wnull-dereference)
if(NEURON_WERROR)
  target_compile_options(neuron_options INTERFACE -Werror)
endif()

# SIMD functions carry their own target attributes. These compatibility options
# never enable host-specific instructions in the rest of the executable.
if(NEURON_NATIVE OR NEURON_AVX2 OR NEURON_AVX512)
  message(STATUS "SIMD kernels use runtime CPUID dispatch; baseline code stays portable")
endif()

if(NEURON_SANITIZE)
  target_compile_options(neuron_options INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(neuron_options INTERFACE -fsanitize=address,undefined)
endif()

configure_file(include/neuron/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/neuron/config.hpp @ONLY)

add_library(neuron
  src/chat.cpp
  src/gguf.cpp
  src/kernels.cpp
  src/dispatch.cpp
  src/kernels_avx2.cpp
  src/model.cpp
  src/quant.cpp
  src/perplexity.cpp
  src/tokenizer.cpp
  src/tensor.cpp
  src/version.cpp)
target_include_directories(neuron PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(neuron PRIVATE neuron_options)

add_executable(neuron_cli src/main.cpp)
set_target_properties(neuron_cli PROPERTIES OUTPUT_NAME neuron)
target_link_libraries(neuron_cli PRIVATE neuron neuron_options)

if(NEURON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
