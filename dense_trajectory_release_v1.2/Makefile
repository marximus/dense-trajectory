#############################################################################
# PREFIX is the prefix to a conda environment and PYTHON_VER is the python
# version used in the conda environment.
# PREFIX and PYTHON_VER are specific to my machine and need to be changed by the
# user compiling. They are left as an example.
#############################################################################
PREFIX=/home/mam588/miniconda3/envs/cilia
PYTHON_VER=3.5m


# set the binaries that have to be built
# TARGETS := DenseTrack Video
TARGETS := DenseTrack.so

# set the build configuration set 
BUILD := release
#BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

# libraries 
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := \
	opencv_core opencv_highgui opencv_videoio opencv_imgproc \
	avformat avdevice avutil avcodec swscale \
	python${PYTHON_VER}

# set some flags and compiler/linker specific commands
CXXFLAGS = -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/opt/include \
		   -I${PREFIX}/include \
		   -I${PREFIX}/include/python${PYTHON_VER}
CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -L/opt/lib -L${PREFIX}/lib -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
