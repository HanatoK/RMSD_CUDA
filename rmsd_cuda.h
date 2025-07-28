#ifndef RMSD_CUDA_H
#define RMSD_CUDA_H

// #define DEBUG

#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nvtx3/nvToolsExt.h>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

typedef double3 AtomPosition;

template <typename T>
class CudaHostAllocator {
public:
  using value_type = T;

  CudaHostAllocator() = default;

  template<typename U>
  constexpr CudaHostAllocator(const CudaHostAllocator<U>&) noexcept {}

  friend bool operator==(const CudaHostAllocator&, const CudaHostAllocator&) { return true; }
  friend bool operator!=(const CudaHostAllocator&, const CudaHostAllocator&) { return false; }

  T* allocate(size_t n) {
    T* ptr;
    if (cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocMapped) != cudaSuccess) {
      // std::cerr << "BAD ALLOC!" << std::endl;
      throw std::bad_alloc();
    }
    // std::cerr << "CudaHostAllocator: allocate at " << ptr << std::endl;
    return ptr;
  }
  void deallocate(T* ptr, size_t n) noexcept {
    cudaFreeHost(ptr);
  }
  template<typename U, typename... Args>
  void construct(U* p, Args&&... args) {
      new(p) U(std::forward<Args>(args)...);
  }

  template<typename U>
  void destroy(U* p) noexcept {
      p->~U();
  }
};

template <typename T>
using host_vector = std::vector<T, CudaHostAllocator<T>>;

bool isDevicePointer(const void* ptr);

class OptimalRotation {
public:
    OptimalRotation(const size_t num_atoms);
    OptimalRotation(const OptimalRotation& rot) = delete; // non-copyable
    ~OptimalRotation();
    void updateReference(const host_vector<AtomPosition>& reference_positions);
    void updateAtoms(const host_vector<AtomPosition>& atom_positions);
//     void bringToCenterHost(AtomPosition* atom_positions, const size_t num_atoms) const;
#if defined (USE_CUDA_GRAPH)
    void bringToCenterDevice(
      AtomPosition* device_atom_positions,
      const size_t num_atoms,
      AtomPosition* center_out,
      unsigned int* counter,
      std::vector<cudaGraphNode_t> dependencies,
      cudaGraphNode_t& last_node);
#else
    void bringToCenterDevice(AtomPosition* device_atom_positions, const size_t num_atoms);
#endif
    void calculateOptimalRotationMatrix(); // main computing function
//     void rotate(AtomPosition* atom_positions, const size_t num_atoms);
#if defined (USE_CUDA_GRAPH)
    void resetGraph();
    double minimalRMSD();
    // double minimalRMSD(const host_vector<AtomPosition>& atom_positions);
#else
    double minimalRMSD() const;
    double minimalRMSD(const host_vector<AtomPosition>& atom_positions) const;
#endif
//     double minimalRMSD(const AtomPosition* atom_positions, const AtomPosition* reference_positions, const size_t num_atoms);
private:
    size_t m_num_atoms;
    AtomPosition* m_device_atom_positions;
    AtomPosition* m_device_reference_positions;
    double* m_device_rotation_matrix; // 3x3
    double* m_device_eigenvalues; // 4
    double* m_device_eigenvectors; // 4x4
    cudaStream_t m_stream;
    // block size of all CUDA kernels
    static const int block_size = 128;
    // buffer for compute center
#if defined (USE_CUDA_GRAPH)
    AtomPosition* m_center_tmp_ref;
    AtomPosition* m_center_tmp_pos;
#else
    AtomPosition* m_center_tmp;
#endif
    // CUDA eigensolver
    cusolverDnHandle_t cusolverH;
    cusolverEigMode_t jobz;
    cublasFillMode_t uplo;
    cusolverStatus_t cusolver_status;
    double* device_work;
    int* devInfo;
    int lwork;
    syevjInfo_t syevj_info;
    // RMSD result
    double* m_host_rmsd;
    double* m_device_rmsd;
#if defined (USE_CUDA_GRAPH)
    unsigned int* d_count_ref;
    unsigned int* d_count_pos;
#endif
    unsigned int* d_count;
#if defined (USE_NR)
    int* max_reached;
#endif
    nvtxEventAttributes_t mEventAttrib;
#if defined (USE_CUDA_GRAPH)
    cudaGraph_t m_graph;
    cudaGraphExec_t m_instance;
    // cudaGraphNode_t last_node;
    struct {
      cudaGraphNode_t updateReferenceNode;
      cudaGraphNode_t updateAtomsNode;
      cudaGraphNode_t centerReferenceNode;
      cudaGraphNode_t centerAtomsNode;
      cudaGraphNode_t buildMatrixFNode;
      cudaGraphNode_t jacobi4x4Node;
      cudaGraphNode_t buildRotationMatrixKernelNode;
    } gpu_nodes;
    bool graphCreated;
#endif
};

#endif
