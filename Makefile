NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -ccbin /usr/bin/g++-7 -Xcompiler -Wall
CXX = /usr/bin/g++-7
CXXFLAGS = -g
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcusolver

all: rmsd_cuda

rmsd_cuda: main.o rmsd_cuda.o rmsd_cuda_kernel.o
	$(CXX) $(LDFLAGS) $(CXXFLAGS) $^ -o $@

main.o: main.cpp rmsd_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

rmsd_cuda.o: rmsd_cuda.cu rmsd_cuda.h rmsd_cuda_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

rmsd_cuda_kernel.o: rmsd_cuda_kernel.cu rmsd_cuda_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf main.o rmsd_cuda.o rmsd_cuda_kernel.o rmsd_cuda
