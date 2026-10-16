cmake_minimum_required(VERSION 3.26)

project(GLM5.3flaash LANGUAGES CXX)

if(NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Default CUDA archs: sm_89 (RTX 4070) + sm_86 fallback.
set(GLM5_CUDA_ARCHS "89;86" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES "${GLM5_CUDA_ARCHS}" CACHE STRING "" FORCE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CPU SIMD level: AVX2 (default), AVX512, or NATIVE.
set(GLM5_SIMD "AVX2" CACHE STRING "CPU SIMD level: AVX2, AVX512, NATIVE")

if(MSVC)
    # Host (C++) options only -- never leak raw MSVC flags into nvcc invocations.
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    if(GLM5_SIMD STREQUAL "AVX512")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
    if(GLM5_SIMD STREQUAL "AVX512")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx512f>)
    elseif(GLM5_SIMD STREQUAL "NATIVE")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    endif()
endif()

enable_testing()

add_subdirectory(cpp)
