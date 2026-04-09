include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(daqp
    GIT_REPOSITORY "https://github.com/darnstrom/daqp"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(daqp)

FetchContent_Declare(lexls
    GIT_REPOSITORY "https://github.com/jrl-umi3218/lexls"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(lexls)

# Disable NIPM tests to avoid build issues
set(COMPILE_TESTS OFF CACHE BOOL "Disable NIPM tests" FORCE)

FetchContent_Declare(nipm
    GIT_REPOSITORY "https://github.com/pfeiffer-kai/NIPM-HLSP"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(nipm)


project(paper_daqp_hierarchy_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(00_basic 00_basic.cpp)
add_executable(random_experiments random_experiments.cpp)
add_executable(mpc_experiments mpc_experiments.cpp)

set(TARGETS 
    00_basic
    random_experiments
    mpc_experiments
)

foreach(TARGET ${TARGETS})
	target_link_libraries(${TARGET} Eigen3::Eigen daqp_eigen lexls nipm_hlsp)
	target_include_directories(
		${TARGET}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
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
