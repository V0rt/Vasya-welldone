cmake_minimum_required(VERSION 3.8)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7/)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

project(locacoca VERSION 1.0.0.0 LANGUAGES CXX CUDA)

#find_package(UHD 3.10.0 REQUIRED)
find_package(Boost COMPONENTS system filesystem thread program_options regex REQUIRED)
find_package(CUDA REQUIRED)
#message("UHD paths: \n"
#        "\t" ${UHD_LIBRARIES} "\n"
#        "\t" ${UHD_INCLUDE_DIRS} "\n"
#        )

#option(UHD_USE_STATIC_LIBS ON)
#set(UHD_BOOST_REQUIRED_COMPONENTS
#        program_options
#        system thread)

add_executable(${PROJECT_NAME}
        main.cu
#        simpleCUFFT.cu
        )

target_link_libraries(${PROJECT_NAME}
        ${UHD_LIBRARIES}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_THREAD_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        )

include_directories(
        helpers
#        ${CMAKE_SOURCE_DIR/helpers}
        ${UHD_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        /usr/local/cuda-11.7/targets/x86_64-linux/include
)

message("CUDA paths: \n"
        "\t" ${CUDA_INCLUDE_DIRS} "\n"
        "\t" ${CUDA_LIBRARIES} "\n"
        "\t" ${CUDA_CUFFT_LIBRARIES} "\n"
        )


#OTHER CUDA LIBS
#[[target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cupti_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_npp_LIBRARY}
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppi_LIBRARY}
        ${CUDA_npps_LIBRARY}
        ${CUDA_nvcuvenc_LIBRARY}
        ${CUDA_nvcuvid_LIBRARY})]]