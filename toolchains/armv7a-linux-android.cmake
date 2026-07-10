# Android NDK toolchain for 32-bit ARM (armv7a)
#
# Usage:
#   cmake .. \
#     -DCMAKE_TOOLCHAIN_FILE=../toolchains/armv7a-linux-android.cmake \
#     -DANDROID_NDK=/path/to/android-ndk-r26

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21)
set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
set(CMAKE_ANDROID_ARM_NEON TRUE)

if(NOT ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK_HOME})
        set(ANDROID_NDK $ENV{ANDROID_NDK_HOME})
    elseif(DEFINED ENV{ANDROID_NDK})
        set(ANDROID_NDK $ENV{ANDROID_NDK})
    else()
        message(FATAL_ERROR "ANDROID_NDK not set. Pass -DANDROID_NDK=/path/to/ndk or set ANDROID_NDK_HOME env var.")
    endif()
endif()

# Use NDK's built-in toolchain file
include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)

# Override for armv7a with NEON
set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
set(CMAKE_ANDROID_ARM_MODE 1)
set(CMAKE_ANDROID_ARM_NEON TRUE)

# Use same flags as host build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mfpu=neon -mfloat-abi=softfp -flto")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mfpu=neon -mfloat-abi=softfp -flto")

# Find zstd from NDK or system
find_library(ZSTD_LIBRARY zstd PATHS ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/arm-linux-androideabi/21 NO_DEFAULT_PATH)
if(NOT ZSTD_LIBRARY)
    message(WARNING "zstd not found in NDK. Static linking may fail. Install zstd for the NDK target.")
endif()

message(STATUS "Android NDK toolchain: ${ANDROID_NDK}")
message(STATUS "Target: armv7a with NEON (32-bit)")
