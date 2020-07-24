#ifndef RMSD_CUDA_H
#define RMSD_CUDA_H

// #define DEBUG

#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef double3 AtomPosition;

bool isDevicePointer(const void* ptr);

class OptimalRotation {
public:
    OptimalRotation(const std::vector<AtomPosition>& atom_positions, const std::vector<AtomPosition>& reference_positions);
    OptimalRotation(const AtomPosition* atom_positions, const AtomPosition* reference_positions, const size_t num_atoms);
    OptimalRotation(const size_t num_atoms);
    OptimalRotation(const OptimalRotation& rot) = delete; // non-copyable
    ~OptimalRotation();
    void updateReference(const std::vector<AtomPosition>& reference_positions);
    void updateAtoms(const std::vector<AtomPosition>& atom_positions);
//     void bringToCenterHost(AtomPosition* atom_positions, const size_t num_atoms) const;
    void bringToCenterDevice(AtomPosition* device_atom_positions, const size_t num_atoms);
    void calculateOptimalRotationMatrix(); // main computing function
//     void rotate(AtomPosition* atom_positions, const size_t num_atoms);
    double minimalRMSD() const;
    double minimalRMSD(const std::vector<AtomPosition>& atom_positions) const;
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
    static const int block_size = 32;
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
    // RMSD result
    double* m_host_rmsd;
    double* m_device_rmsd;
};

#endif
