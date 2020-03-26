cmake_minimum_required (VERSION 2.8.12)
project(mcmc CXX)

if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # MACOSX_RPATH is enabled by default
endif()

set(CMAKE_MACOSX_RPATH ON)


# Set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/enoki/include
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tbb/include
  ${CMAKE_CURRENT_SOURCE_DIR}/pcg32
)


include (CheckCXXCompilerFlag)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  # Enable C++17
  CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)
  if (NOT HAS_CPP17_FLAG)
    message(FATAL_ERROR "Unsupported compiler -- C++17 support is needed!")
  endif()
endif()

# Compile with compiler warnings turned on
if (MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    add_compile_options("/W4")
  endif()
else()
  add_compile_options("-Wall" "-Wextra")
endif()

include(CheckCXXSourceRuns)

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }" ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

# Prefer libc++ in conjunction with Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (HAS_LIBCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
    message(STATUS "Enoki: using libc++.")
  else()
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP_AND_CPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
    if (HAS_LIBCPP_AND_CPPABI)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      message(STATUS "Enoki: using libc++ and libc++abi.")
    else()
      message(FATAL_ERROR "When Clang is used to compile Enoki, libc++ must be available -- GCC's libstdc++ is not supported! (please insteall the libc++ development headers, provided e.g. by the packages 'libc++-dev' and 'libc++abi-dev' on Debian/Ubuntu).")
    endif()
  endif()
endif()


#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++17")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
  #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
#endif()

#set(PYTHON_EXECUTABLE "/Users/af/anaconda3/bin/python3.6" CACHE FILEPATH "python filepath" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tbb)

set(TBB_BUILD_SHARED          ON  CACHE BOOL " " FORCE)
set(TBB_BUILD_STATIC          ON CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC       ON CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY ON CACHE BOOL " " FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/enoki)

enoki_set_native_flags()
enoki_set_compile_flags()
#add_compile_options("-march=native")
#add_compile_options("-fno-stack-protector")
#add_compile_options("-fomit-frame-pointer")
#add_compile_options("-O1" "-DDOUBLE")

add_compile_options("-Wno-unused-parameter")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif (MSVC)
  add_compile_options("/std:c++17")
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable (mcmc) 
target_sources(mcmc
  PRIVATE 
    test.cpp
) 
set_target_properties(mcmc PROPERTIES COMPILE_DEFINITIONS "PY=0")
target_link_libraries(mcmc PRIVATE tbb)



add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
pybind11_add_module(mcmcpy test.cpp)
set_target_properties(mcmcpy PROPERTIES COMPILE_DEFINITIONS "PY=1")
target_link_libraries(mcmcpy PRIVATE tbb)
set_target_properties(mcmcpy PROPERTIES OUTPUT_NAME mcmc)
