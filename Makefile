###########################################################

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda
JACOBIDIR = Cuda/jacobi
QRDIR = Cuda/QR
TIMEDIR= utils/Time
HELPFILESDIR = utils/Matrix_helper
MYLAPACKELIB = lapack

##########################################################

## CC COMPILER OPTIONS ##
CC=gcc
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##
NVCC=nvcc
NVCC_FLAGS=-arch=sm_60
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lstdc++

##########################################################

## Make variables ##

# Target executable name:
EXE = eigenvalues.x

# Object files:
OBJS = main.o ./$(JACOBIDIR)/jacobiSerial.o ./$(TIMEDIR)/time.o ./$(HELPFILESDIR)/matrix_generate_random.o ./$(HELPFILESDIR)/printResults.o ./$(MYLAPACKELIB)/manageRoutines.o -llapacke -lm ./$(JACOBIDIR)/jacobiParallel.o ./$(QRDIR)/QRSymetricParallel.o
CUDA_OBJS_1 = ./$(JACOBIDIR)/jacobiParallel.o
CUDA_OBJS_2 = ./$(QRDIR)/QRSymetricParallel.o

##########################################################

## Compile ##
# Link c and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile CUDA source files to object files:
$(CUDA_OBJS_1) : ./$(JACOBIDIR)/jacobiParallel.cu ./$(JACOBIDIR)/jacobiParallel.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Compile CUDA source files to object files: 
#TODO: make one compiler for every cuda object
$(CUDA_OBJS_2) : ./$(QRDIR)/QRSymetricParallel.cu ./$(QRDIR)/QRSymetricParallel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)