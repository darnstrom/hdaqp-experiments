include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(EIGEN ON CACHE BOOL "Enable DAQP Eigen interface" FORCE)

FetchContent_Declare(daqp
    GIT_REPOSITORY "https://github.com/darnstrom/daqp"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(daqp)

FetchContent_Declare(lexls
    GIT_REPOSITORY "https://github.com/jrl-umi3218/lexls"
    GIT_TAG "origin/master"
)
set(INSTALL_HTML_DOCUMENTATION OFF CACHE BOOL "Disable LexLS HTML docs" FORCE)
set(INSTALL_PDF_DOCUMENTATION OFF CACHE BOOL "Disable LexLS PDF docs" FORCE)
FetchContent_MakeAvailable(lexls)

# Disable NIPM tests to avoid build issues
set(COMPILE_TESTS OFF CACHE BOOL "Disable NIPM tests" FORCE)

FetchContent_Declare(nipm
    GIT_REPOSITORY "https://github.com/pfeiffer-kai/NIPM-HLSP"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(nipm)

FetchContent_Declare(highs
    GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS"
    GIT_TAG "origin/master"
)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable HiGHS examples" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable HiGHS tests" FORCE)
FetchContent_MakeAvailable(highs)


project(paper_daqp_hierarchy_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(HDF5 QUIET COMPONENTS CXX)
if(NOT HDF5_FOUND)
    find_path(HDF5_INCLUDE_DIRS
        NAMES H5Cpp.h
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES include/hdf5/serial include
    )
    find_library(HDF5_SERIAL_CPP_LIBRARY
        NAMES hdf5_serial_cpp hdf5_cpp
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )
    find_library(HDF5_SERIAL_HL_CPP_LIBRARY
        NAMES hdf5_serial_hl_cpp hdf5_hl_cpp
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )
    find_library(HDF5_SERIAL_HL_LIBRARY
        NAMES hdf5_serial_hl hdf5_hl
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )
    find_library(HDF5_SERIAL_LIBRARY
        NAMES hdf5_serial hdf5
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )

    if(NOT HDF5_INCLUDE_DIRS OR
       NOT HDF5_SERIAL_CPP_LIBRARY OR
       NOT HDF5_SERIAL_HL_CPP_LIBRARY OR
       NOT HDF5_SERIAL_HL_LIBRARY OR
       NOT HDF5_SERIAL_LIBRARY)
        message(FATAL_ERROR "HDF5 C++ headers/libraries not found")
    endif()

    set(HDF5_CXX_LIBRARIES
        ${HDF5_SERIAL_CPP_LIBRARY}
        ${HDF5_SERIAL_HL_CPP_LIBRARY}
        ${HDF5_SERIAL_HL_LIBRARY}
        ${HDF5_SERIAL_LIBRARY}
    )
    set(HDF5_LIBRARIES ${HDF5_CXX_LIBRARIES})
endif()

add_executable(00_basic 00_basic.cpp)
add_executable(random_experiments random_experiments.cpp)
add_executable(mpc_experiments mpc_experiments.cpp)

set(TARGETS 
    00_basic
    random_experiments
    mpc_experiments
)

foreach(TARGET ${TARGETS})
	target_link_libraries(${TARGET} Eigen3::Eigen daqp_eigen lexls nipm_hlsp highs)
	target_include_directories(
		${TARGET}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
			${daqp_SOURCE_DIR}/include
			${daqp_SOURCE_DIR}/interfaces/daqp-eigen
			${nipm_SOURCE_DIR}
	)
endforeach()

target_link_libraries(mpc_experiments ${HDF5_CXX_LIBRARIES})
target_include_directories(mpc_experiments PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_features(mpc_experiments PRIVATE cxx_std_17)
target_compile_definitions(
    mpc_experiments
    PRIVATE
        HDAQP_EXPERIMENTS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
