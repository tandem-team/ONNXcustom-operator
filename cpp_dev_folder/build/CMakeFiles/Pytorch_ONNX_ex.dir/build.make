# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build

# Include any dependencies generated for this target.
include CMakeFiles/Pytorch_ONNX_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Pytorch_ONNX_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Pytorch_ONNX_ex.dir/flags.make

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o: CMakeFiles/Pytorch_ONNX_ex.dir/flags.make
CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o: ../rlcustom_operator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o -c /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/rlcustom_operator.cpp

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/rlcustom_operator.cpp > CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.i

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/rlcustom_operator.cpp -o CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.s

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.requires:

.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.requires

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.provides: CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.requires
	$(MAKE) -f CMakeFiles/Pytorch_ONNX_ex.dir/build.make CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.provides.build
.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.provides

CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.provides.build: CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o


# Object files for target Pytorch_ONNX_ex
Pytorch_ONNX_ex_OBJECTS = \
"CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o"

# External object files for target Pytorch_ONNX_ex
Pytorch_ONNX_ex_EXTERNAL_OBJECTS =

Pytorch_ONNX_ex: CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o
Pytorch_ONNX_ex: CMakeFiles/Pytorch_ONNX_ex.dir/build.make
Pytorch_ONNX_ex: /home/tandem-team/.local/lib/python3.6/site-packages/torch/lib/libtorch.so
Pytorch_ONNX_ex: /home/tandem-team/.local/lib/python3.6/site-packages/torch/lib/libc10.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/stubs/libcuda.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libnvrtc.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libnvToolsExt.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libcudart.so
Pytorch_ONNX_ex: /home/tandem-team/.local/lib/python3.6/site-packages/torch/lib/libc10_cuda.so
Pytorch_ONNX_ex: /home/tandem-team/.local/lib/python3.6/site-packages/torch/lib/libc10_cuda.so
Pytorch_ONNX_ex: /home/tandem-team/.local/lib/python3.6/site-packages/torch/lib/libc10.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libcufft.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libcurand.so
Pytorch_ONNX_ex: /usr/lib/aarch64-linux-gnu/libcublas.so
Pytorch_ONNX_ex: /usr/lib/aarch64-linux-gnu/libcudnn.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libnvToolsExt.so
Pytorch_ONNX_ex: /usr/local/cuda/lib64/libcudart.so
Pytorch_ONNX_ex: CMakeFiles/Pytorch_ONNX_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Pytorch_ONNX_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Pytorch_ONNX_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Pytorch_ONNX_ex.dir/build: Pytorch_ONNX_ex

.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/build

CMakeFiles/Pytorch_ONNX_ex.dir/requires: CMakeFiles/Pytorch_ONNX_ex.dir/rlcustom_operator.cpp.o.requires

.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/requires

CMakeFiles/Pytorch_ONNX_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Pytorch_ONNX_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/clean

CMakeFiles/Pytorch_ONNX_ex.dir/depend:
	cd /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build /home/tandem-team/Work_Folder/Pytorch_ONNX_ex/cpp_dev_folder/build/CMakeFiles/Pytorch_ONNX_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Pytorch_ONNX_ex.dir/depend

