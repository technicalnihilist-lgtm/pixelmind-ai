# Optional accelerator layer.
#
# Not wired into the default build: the Kotlin scheduler in LcmScheduler.kt produces identical
# results, so the project builds (and ships) without the NDK. To enable, see docs/BUILD.md:
# uncomment the externalNativeBuild block in app/build.gradle.kts and point it here.
cmake_minimum_required(VERSION 3.22.1)
project(pixelmind CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pixelmind SHARED pixelmind_native.cpp)

target_compile_options(pixelmind PRIVATE
        -O3
        -ffast-math
        -fvisibility=hidden)

# arm64-v8a is the only ABI the app ships; NEON is baseline there.
if (ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(pixelmind PRIVATE -march=armv8-a+simd)
endif()

target_link_libraries(pixelmind PRIVATE log)
