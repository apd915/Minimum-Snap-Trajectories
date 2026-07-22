cmake_minimum_required(VERSION 3.16)
project(MinimumSnapEngine VERSION 1.0 LANGUAGES CXX)

# Force modern C++ standards (Highly recommended for Eigen3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. FIND REQUIRED LIBRARIES
# ==========================================
# Because you installed these natively via apt, CMake will automatically 
# search your system's /usr/include directories for them.
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(osqp REQUIRED)

# ==========================================
# 2. BUILD THE MATH ENGINE LIBRARY
# ==========================================
# We compile your B-spline math into a SHARED library (.so). 
# This is crucial so your Python scripts can import it later!
add_library(min_snap_engine SHARED
    src/min_snap_natural.cpp
    src/min_snap_clamped.cpp
    src/matrix_builder.cpp
    src/optimize.cpp
    src/astar_sfc.cpp
    src/voxel_grid.cpp
    src/static_sfc.cpp
    src/dynamic_sfc.cpp
    src/front_end.cpp
    src/trajectory_planner.cpp
)

# Tell the compiler where to find your .hpp header files
target_include_directories(min_snap_engine PUBLIC include)

# Link the Eigen3 math library to your engine
target_link_libraries(min_snap_engine PUBLIC Eigen3::Eigen osqp::osqp)

# ==========================================
# 3. BUILD THE GOOGLE TEST SUITE
# ==========================================
enable_testing()

add_executable(bspline_tests
    tests/test_min_snap.cpp
    tests/test_optimize.cpp
    tests/test_astar.cpp
    tests/test_voxel_grid.cpp
    tests/test_sfc.cpp
    tests/test_front_end.cpp
    tests/test_trajectory_planner.cpp
)

# Link the test suite to your engine AND to Google Test's main runner
target_link_libraries(bspline_tests
    PRIVATE
    min_snap_engine
    GTest::gtest_main
)

# Make CTest aware of the Google Tests
include(GoogleTest)
gtest_discover_tests(bspline_tests)

# ==========================================
# 4. PYTHON BINDINGS (PYBIND11)
# ==========================================
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(min_snap_bindings src/python_bindings.cpp)
target_include_directories(min_snap_bindings PUBLIC include)
target_link_libraries(min_snap_bindings PRIVATE min_snap_engine)