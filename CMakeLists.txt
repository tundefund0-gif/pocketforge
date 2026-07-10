cmake_minimum_required(VERSION 3.20)
project(PocketForge VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Check for NEON
include(CheckCSourceCompiles)
check_c_source_compiles("
int main() {
    #ifdef __ARM_NEON
    return 0;
    #else
    #error no neon
    #endif
}" HAS_ARM_NEON)

check_c_source_compiles("
int main() {
    #ifdef __ARM_NEON
    #ifdef __aarch64__
    return 0;
    #else
    #error no aarch64
    #endif
    #else
    #error no neon
    #endif
}" HAS_AARCH64_NEON)

if(HAS_AARCH64_NEON)
    message(STATUS "ARM64 NEON detected — enabling SIMD paths")
    add_compile_definitions(FORGE_ARM64_NEON=1)
    add_compile_options(-O3 -mcpu=native)
elseif(HAS_ARM_NEON)
    message(STATUS "ARM32 NEON detected — enabling SIMD paths")
    add_compile_definitions(FORGE_ARM32_NEON=1)
    add_compile_options(-O3 -mfpu=neon)
else()
    message(STATUS "No NEON detected — using scalar fallback")
    add_compile_options(-O3 -march=native)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -flto)

# Core library
file(GLOB FORGE_SRC src/*.cpp)
list(REMOVE_ITEM FORGE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM FORGE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/quant_tool.cpp")

add_library(pocketforge STATIC ${FORGE_SRC})
target_include_directories(pocketforge PUBLIC include)
target_link_libraries(pocketforge pthread zstd)

# Quantizer tool
add_executable(forge-quant src/quant_tool.cpp)
target_link_libraries(forge-quant pocketforge pthread zstd)

# Main CLI
add_executable(forge src/main.cpp)
target_link_libraries(forge pocketforge pthread zstd)

# API Server
add_executable(forge-server src/forge_server.cpp src/http_server.cpp)
target_link_libraries(forge-server pocketforge pthread zstd)

# Tests
enable_testing()
add_subdirectory(tests)
