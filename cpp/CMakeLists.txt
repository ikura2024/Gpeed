# Core runtime library (CPU). CUDA kernels are added under kernels/ when enabled.
add_library(glm5_core STATIC
    core/src/gguf.cpp
    core/src/quant.cpp
    core/src/model_config.cpp
    core/src/weights.cpp
    core/src/model.cpp
    core/src/model_batch.cpp
    core/src/kv_cache.cpp
    core/src/indexer.cpp
    core/src/tokenizer.cpp
)

target_include_directories(glm5_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

if(MSVC)
    target_compile_options(glm5_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
    # AVX2/FMA matvec kernels (i7-13700F and any modern x86-64).
    target_compile_options(glm5_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(glm5_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

target_compile_definitions(glm5_core PUBLIC
    GLM5_VERSION="${PROJECT_VERSION}"
)

# --- tools ---------------------------------------------------------------
add_executable(gguf_dump tools/gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE glm5_core)

add_executable(gguf_dequant tools/gguf_dequant.cpp)
target_link_libraries(gguf_dequant PRIVATE glm5_core)

add_executable(glm5_generate tools/glm5_generate.cpp)
target_link_libraries(glm5_generate PRIVATE glm5_core)

add_executable(glm5_bench tools/glm5_bench.cpp)
target_link_libraries(glm5_bench PRIVATE glm5_core)

# --- tests ---------------------------------------------------------------
add_executable(gguf_selftest ${CMAKE_SOURCE_DIR}/tests/gguf_selftest.cpp)
target_link_libraries(gguf_selftest PRIVATE glm5_core)

add_executable(glm5_selftest ${CMAKE_SOURCE_DIR}/tests/glm5_selftest.cpp)
target_link_libraries(glm5_selftest PRIVATE glm5_core)
add_test(NAME glm5_selftest COMMAND glm5_selftest)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(GLM5_TEST_DIR ${CMAKE_BINARY_DIR}/test)
    add_custom_command(
        OUTPUT ${GLM5_TEST_DIR}/test.gguf ${GLM5_TEST_DIR}/test.expected.txt
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/python/tools/make_test_gguf.py --out ${GLM5_TEST_DIR}
        DEPENDS ${CMAKE_SOURCE_DIR}/python/tools/make_test_gguf.py
                ${CMAKE_SOURCE_DIR}/cpp/core/include/glm5/iq3xxs_tables.h
        COMMENT "Generating synthetic GGUF for tests"
    )
    add_custom_target(gen_test_gguf ALL
        DEPENDS ${GLM5_TEST_DIR}/test.gguf ${GLM5_TEST_DIR}/test.expected.txt)
    add_dependencies(gguf_selftest gen_test_gguf)

    add_test(NAME gguf_selftest
             COMMAND gguf_selftest ${GLM5_TEST_DIR}/test.gguf ${GLM5_TEST_DIR}/test.expected.txt)
endif()

# --- CUDA kernels (optional) --------------------------------------------
option(GLM5_ENABLE_CUDA "Build CUDA kernels" ON)
if(GLM5_ENABLE_CUDA)
    add_subdirectory(kernels)
endif()
