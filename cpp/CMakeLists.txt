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

# ==========================================
# 2. BUILD THE MATH ENGINE LIBRARY
# ==========================================
# We compile your B-spline math into a SHARED library (.so). 
# This is crucial so your Python scripts can import it later!
add_library(min_snap_engine SHARED
    src/min_snap_natural.cpp
    src/min_snap_clamped.cpp
)

# Tell the compiler where to find your .hpp header files
target_include_directories(min_snap_engine PUBLIC include)

# Link the Eigen3 math library to your engine
target_link_libraries(min_snap_engine PUBLIC Eigen3::Eigen)

# ==========================================
# 3. BUILD THE GOOGLE TEST SUITE
# ==========================================
enable_testing()

add_executable(bspline_tests
    tests/test_min_snap.cpp
)

# Link the test suite to your engine AND to Google Test's main runner
target_link_libraries(bspline_tests
    PRIVATE
    min_snap_engine
    GTest::gtest_main
)