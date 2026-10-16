# Custom CUDA kernels (no cuBLAS / cuDNN).
add_library(glm5_kernels STATIC
    src/device_info.cu
)

target_include_directories(glm5_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(glm5_kernels PUBLIC CUDA::cudart)

# Host/device interop; separable (RDC) compilation is enabled only when a
# translation unit actually needs device linking.
option(GLM5_CUDA_RDC "Enable CUDA separable compilation" OFF)
if(GLM5_CUDA_RDC)
    set_target_properties(glm5_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_compile_options(glm5_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
