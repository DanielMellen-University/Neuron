function(neuron_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE neuron neuron_options)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

neuron_add_test(test_gguf)
neuron_add_test(test_quant)
neuron_add_test(test_kernels)
neuron_add_test(test_tokenizer)
neuron_add_test(test_unicode)
neuron_add_test(test_model)
neuron_add_test(test_chat)
neuron_add_test(test_shape)
neuron_add_test(test_version)

add_test(NAME cli_version COMMAND neuron_cli --version)
set_tests_properties(cli_version PROPERTIES
  PASS_REGULAR_EXPRESSION "^neuron ${PROJECT_VERSION} \\(")

add_test(NAME cli_rejects_unknown_args COMMAND neuron_cli --bogus)
set_tests_properties(cli_rejects_unknown_args PROPERTIES WILL_FAIL TRUE)

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
add_test(NAME test_harness_compare
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_harness.py)

add_test(NAME test_qwen_fixtures
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_qwen_fixtures.py $<TARGET_FILE:neuron_cli>)

add_test(NAME test_m4_method
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_m4_method.py $<TARGET_FILE:neuron_cli>)

add_executable(test_simd test_simd.cpp)
target_link_libraries(test_simd PRIVATE neuron neuron_options)
foreach(isa IN ITEMS scalar avx2 avx512)
  add_test(NAME simd_${isa} COMMAND test_simd ${isa})
  set_tests_properties(simd_${isa} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
