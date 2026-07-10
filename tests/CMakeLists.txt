include(CTest)

# Test utilities
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Helper to add a test
function(add_forge_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} pocketforge pthread zstd)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 60)
endfunction()

# Unit tests
add_forge_test(test_types)
add_forge_test(test_matmul)
add_forge_test(test_kv_cache)
add_forge_test(test_mtp)
add_forge_test(test_layer_skip)
add_forge_test(test_prefetch)
add_forge_test(test_quant_format)
add_forge_test(test_engine)
add_forge_test(test_memory)
add_forge_test(test_benchmark)
add_forge_test(test_tool_calling)
