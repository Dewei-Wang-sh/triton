add_triton_library(TritonLLVMTransforms
    Vectorize.cpp
    #SimpleSchedule.cpp

    DEPENDS
    TritonLLVMTransformsIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRReconcileUnrealizedCasts
    LLVMCore
    LLVMPasses
    LLVMSupport
)


target_include_directories(TritonLLVMTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonLLVMTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
