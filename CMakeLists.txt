cmake_minimum_required(VERSION 3.0)
project(OpenCV_tests)

# Require C++11 or later
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")

    message([main] " OpenMP_C_FLAGS=${OpenMP_C_FLAGS}") # -fopenmp
    message([main] " OpenMP_CXX_FLAGS}=${OpenMP_CXX_FLAGS}") # -fopenmp
    message([main] " OpenMP_EXE_LINKER_FLAGS=${OpenMP_EXE_LINKER_FLAGS}") # ***

    # no use for xxx_INCLUDE_DIRS and xxx_libraries for OpenMP
    message([main] " OpenMP_INCLUDE_DIRS=${OpenMP_INCLUDE_DIRS}") # ***
    message([main] " OpenMP_LIBRARIES=${OpenMP_LIBRARIES}") # ***

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
# set(SOURCES parallel_orb.cpp other_source.cpp)
# set(SOURCES parallel_orb.cpp)

# Create executable
add_executable(orb_parallel src/parallel_orb.cpp)
# Link against OpenCV libraries
target_link_libraries(orb_parallel ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Create executable
add_executable(orb_extraction src/orb_extraction.cpp)
# Link against OpenCV libraries
target_link_libraries(orb_extraction ${OpenCV_LIBS})
