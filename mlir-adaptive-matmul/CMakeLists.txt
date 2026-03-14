cmake_minimum_required(VERSION 3.10)
project(mlir-adaptive-matmul)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

set(MLIR_DIR "/home/sumit/mlir-build/lib/cmake/mlir")
set(LLVM_DIR "/home/sumit/mlir-build/lib/cmake/llvm")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
  ${MLIR_LIBRARY_DIRS}
  ${LLVM_LIBRARY_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})

# Header files
set(ADAPTIVE_MATMUL_HEADERS
  include/AdaptiveMatmul/CostModel.h
  include/AdaptiveMatmul/MatrixAnalyzer.h
  include/AdaptiveMatmul/HardwareInfo.h
  include/AdaptiveMatmul/Router.h
)

# Source files
set(ADAPTIVE_MATMUL_SOURCES
  src/analysis/MatrixAnalyzer.cpp
  src/router/RouterPass.cpp
  src/cost_model/CostModel.cpp
  src/hardware/HardwareInfo.cpp
  src/skinny_passes/SkinnyTilingPass.cpp
  src/skinny_passes/SkinnyVectorizationPass.cpp
  src/skinny_passes/SkinnyMemoryPass.cpp
  src/generic_passes/GenericTilingPass.cpp
  src/generic_passes/WideTilingPass.cpp
  src/generic_passes/SquareBlockingPass.cpp
  src/generic_passes/SIMDVectorizationPass.cpp
  src/generic_passes/GPUOffloadPass.cpp
  src/generic_passes/KernelFusionPass.cpp
  src/small_passes/SmallMatrixVectorPass.cpp
  src/lowering/LowerToLLVMPass.cpp
  src/jit/AdaptiveJITRunner.cpp
)

add_library(AdaptiveMatmul STATIC ${ADAPTIVE_MATMUL_SOURCES} ${ADAPTIVE_MATMUL_HEADERS})

target_link_libraries(AdaptiveMatmul
  PUBLIC
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRTilingInterface
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRVectorToLLVM
  MLIRVectorToSCF
  MLIRMathToLLVM
  MLIRUBToLLVM
  MLIRAffineToStandard
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRConvertToLLVMPass
  MLIRUBDialect
  MLIRLLVMDialect
  MLIRExecutionEngine
  MLIRExecutionEngineUtils
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRPass
  MLIRTransforms
  MLIRIR
  LLVMSupport
  LLVMCore
  LLVMOrcJIT
  LLVMOrcShared
  LLVMOrcTargetProcess
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Desc
  LLVMX86Info
)

# Custom mlir-opt like tool
add_executable(adaptive-opt
  src/main.cpp
)

target_link_libraries(adaptive-opt
  PRIVATE
  AdaptiveMatmul
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  MLIRParser
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
  LLVMSupport
  LLVMCore
)
