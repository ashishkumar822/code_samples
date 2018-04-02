cmake_minimum_required(VERSION 2.8.0)



set(CURR_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB all_files RELATIVE ${CURR_DIR} ${CURR_DIR}/*)

message(STATUS ${all_files})

find_package(Boost REQUIRED filesystem system timer chrono )

set(SUBDIRS)

foreach(dir ${all_files})
if(IS_DIRECTORY ${CURR_DIR}/${dir})
list(APPEND SUBDIRS ${dir})
endif()
endforeach()

message(STATUS ${SUBDIRS})

foreach(dir ${SUBDIRS})
set(CURRENT_TARGET ${dir})
add_subdirectory(${CURR_DIR}/${dir})
endforeach()

