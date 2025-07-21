#ifndef RMSD_CUDA_H
#define RMSD_CUDA_H

// #define DEBUG

#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cusolverDn.h>

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
    OptimalRotation(const host_vector<AtomPosition>& atom_positions, const host_vector<AtomPosition>& reference_positions);
    OptimalRotation(const AtomPosition* atom_positions, const AtomPosition* reference_positions, const size_t num_atoms);
    OptimalRotation(const size_t num_atoms);
    OptimalRotation(const OptimalRotation& rot) = delete; // non-copyable
    ~OptimalRotation();
    void updateReference(const host_vector<AtomPosition>& reference_positions);
    void updateAtoms(const host_vector<AtomPosition>& atom_positions);
//     void bringToCenterHost(AtomPosition* atom_positions, const size_t num_atoms) const;
    void bringToCenterDevice(AtomPosition* device_atom_positions, const size_t num_atoms);
    void calculateOptimalRotationMatrix(); // main computing function
//     void rotate(AtomPosition* atom_positions, const size_t num_atoms);
    double minimalRMSD() const;
    double minimalRMSD(const host_vector<AtomPosition>& atom_positions) const;
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
    AtomPosition* m_center_tmp;
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
    unsigned int* d_count;
};

#endif
