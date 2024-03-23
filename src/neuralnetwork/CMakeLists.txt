set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/neuralnetwork/NeuralNetwork.h" "${PROJECT_SOURCE_DIR}/include/neuralnetwork/Layer.h" "${PROJECT_SOURCE_DIR}/include/neuralnetwork/LinearLayer.h" "${PROJECT_SOURCE_DIR}/include/neuralnetwork/ReLULayer.h" "${PROJECT_SOURCE_DIR}/include/neuralnetwork/SigmoidLayer.h")

add_library(neuralnetwork SHARED NeuralNetwork.cu LinearLayer.cu ReLULayer.cu SigmoidLayer.cu ${HEADER_LIST})

target_include_directories(neuralnetwork PUBLIC ../../include)

target_compile_features(neuralnetwork PUBLIC cxx_std_11)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})
