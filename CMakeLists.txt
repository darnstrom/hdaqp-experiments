cmake_minimum_required(VERSION 3.22)

project(paper_daqp_hierarchy VERSION 0.0.1 DESCRIPTION "Complementary material" LANGUAGES CXX)
message(PROJECT_NAME="${PROJECT_NAME}")
message(PROJECT_VERSION="${PROJECT_VERSION}")

include(CTest)
add_subdirectory(tests)
