add_executable(main main.cu)
target_compile_features(main PRIVATE cxx_std_11)

target_link_libraries(main PRIVATE matrix neuralnetwork dataset)
