MAKEFLAGS += -j20 #parallel compilation

# you may need these in your .bashrc
# export CUDA_HOME=/usr/local/cuda-11.0
# export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}$ 
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#files that need compilin'
OBJDIR = ./bin
SRCDIR = src
SRCFILES = $(wildcard $(SRCDIR)/atm/*.cpp) $(wildcard $(SRCDIR)/*.cpp) 

NSRCFILES = $(SRCFILES) generate_source_function.cpp
NSRCFILES := $(filter-out src/observation_fit.cpp, $(NSRCFILES))
NOBJFILES    := $(filter %.o, $(NSRCFILES:%.cpp=$(OBJDIR)/%.cuda.o)       $(NSRCFILES:%.cu=$(OBJDIR)/%.cuda.o      ))
NOBJFILESDBG := $(filter %.o, $(NSRCFILES:%.cpp=$(OBJDIR)/%.cuda.debug.o) $(NSRCFILES:%.cu=$(OBJDIR)/%.cuda.debug.o))

#include directories
BOOSTDIR=-I/home/mike/Documents/Utilities/boost_1_73_0/
EIGENDIR=-I/home/mike/Documents/Utilities/eigen_git/
IDIR=-I$(SRCDIR) $(BOOSTDIR) $(EIGENDIR)

# GNU Compiler
CC=g++
LIBS=-lm 
MPFLAGS=-fopenmp
OFLAGS=-O3 -march=native -DNDEBUG 

# Nvidia CUDA Compiler
NCC=nvcc -Xcudafe --display_error_number #--disable-warnings
NFLAGS=-x cu -D RT_FLOAT              -D EIGEN_NO_CUDA                -D BOOST_NO_CUDA
#            ^^^^32-bit calculation   ^^^^^ disable Eigen on device   ^^^^^ disable Boost on device
#                                           (some modifications to    (added this flag by hand as a wrapper
#                                            Eigen were needed to      around BOOST_GPU_ENABLED in
#                                            get this to work)         boost/config/compiler/nvcc.hpp)
NIDIR=$(IDIR) \
      -L$(CUDA_HOME)/lib64/ \
      -I$(CUDA_HOME)/samples/common/inc/
NLIBS=-lm -lcudart -lcusolver -lcublas
NOBASEFLAGS= -O3 -DNDEBUG -lineinfo #--maxrregcount 64 -arch sm_61 --use_fast_math 
# if we are CUDA 11, link time optimization is possible
ifeq ($(shell cat $$CUDA_HOME/version.txt | awk '{split($$3,a,"."); print a[1]}'),11)
#$(info Using CUDA 11 link time optimization)
NOFLAGS=$(NOBASEFLAGS) -dlto
else
NOFLAGS=$(NOBASEFLAGS)
endif
NDBGFLAGS=-O0 -g -lineinfo


# # intel compiler
# you may need to run this
# source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
ICC=icpc
ICOMPILER_OPT=-std=c++17
ILIBS=-fPIC -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
IMPFLAGS=-qopenmp
IOFLAGS=-DNDEBUG -fast

all: generate_source_function generate_source_function_gpu

generate_source_function:
	$(CC) generate_source_function.cpp $(SRCFILES) $(IDIR) $(LIBS) $(MPFLAGS) $(OFLAGS) -o generate_source_function.x

generate_source_function_intel:
	$(ICC) generate_source_function.cpp $(SRCFILES) $(ICOMPILER_OPT) $(IDIR) $(ILIBS) $(IMPFLAGS) $(IOFLAGS) -o generate_source_function.x

generate_source_function_profile:
	$(CC) generate_source_function.cpp $(SRCFILES) $(IDIR) $(LIBS) $(OFLAGS) -g -o generate_source_function.x

generate_source_function_debug_warn:
	$(CC) generate_source_function.cpp $(SRCFILES) $(IDIR) $(LIBS) -O0 -g -Wall -Wextra -o generate_source_function.x

generate_source_function_float:
	$(CC) -D RT_FLOAT generate_source_function.cpp $(SRCFILES) $(IDIR) $(LIBS) $(MPFLAGS) $(OFLAGS) -o generate_source_function.x



generate_source_function_gpu: $(NOBJFILES) 
	@echo "linking..."
	@$(NCC) $(NOBJFILES) $(NIDIR) $(NLIBS) $(NOFLAGS) -o generate_source_function_gpu.x

$(OBJDIR)/%.cuda.o: %.cpp
	@echo "compiling $<..."
	@mkdir -p '$(@D)'
	@$(NCC) $(NFLAGS) $(NIDIR) $(NLIBS) $(NOFLAGS) -dc $< -o $@

$(OBJDIR)/%.cuda.o: %.cu
	@echo "compiling $<..."
	@mkdir -p '$(@D)'
	@$(NCC) $(NFLAGS) $(NIDIR) $(NLIBS) $(NOFLAGS) -dc $< -o $@




generate_source_function_gpu_debug: $(NOBJFILESDBG) 
	@echo "linking ..."
	@$(NCC) $(NOBJFILESDBG) $(NIDIR) $(NLIBS) $(NDBGFLAGS) -o generate_source_function_gpu.x

$(OBJDIR)/%.cuda.debug.o: %.cpp
	@echo "compiling $<..."
	@mkdir -p '$(@D)'
	@$(NCC) $(NFLAGS) $(NIDIR) $(NLIBS) $(NDBGFLAGS) -dc $< -o $@

$(OBJDIR)/%.cuda.debug.o: %.cu
	@echo "compiling $<..."
	@mkdir -p '$(@D)'
	@$(NCC) $(NFLAGS) $(NIDIR) $(NLIBS) $(NDBGFLAGS) -dc $< -o $@


clean_gpu:
	rm -f generate_source_function_gpu.x
	test -d ./bin && find ./bin/ -type f -name '*cuda*' -delete 2>/dev/null

clean_all:
	rm -f generate_source_function.x
	rm -f generate_source_function_gpu.x
	rm -rf bin
