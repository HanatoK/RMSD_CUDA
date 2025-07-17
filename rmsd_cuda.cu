#include "rmsd_cuda.h"
#include "rmsd_cuda_kernel.cuh"
#include <cusolverDn.h>
#include <iostream>

bool isDevicePointer(const void* ptr) {
    bool is_device_pointer = true;
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "here" << std::endl;
        return false;
    }
    if (attributes.devicePointer) {
        is_device_pointer = true;
    } else {
        is_device_pointer = false;
    }
    std::cout << std::boolalpha << is_device_pointer << '\n';
    return is_device_pointer;
}

OptimalRotation::OptimalRotation(const std::vector<AtomPosition>& atom_positions, const std::vector<AtomPosition>& reference_positions):
OptimalRotation(atom_positions.data(), reference_positions.data(), atom_positions.size())
{}

OptimalRotation::OptimalRotation(const AtomPosition* atom_positions, const AtomPosition* reference_positions, const size_t num_atoms): OptimalRotation(num_atoms) {
    cudaMemcpyAsync(m_device_reference_positions, reference_positions, m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(m_device_atom_positions, atom_positions, m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream);
    bringToCenterDevice(m_device_reference_positions, m_num_atoms);
    bringToCenterDevice(m_device_atom_positions, m_num_atoms);
    calculateOptimalRotationMatrix();
    cudaStreamSynchronize(m_stream);
}

OptimalRotation::OptimalRotation(const size_t num_atoms) {
    cudaStreamCreate(&m_stream);
    m_num_atoms = num_atoms;
    cudaMalloc(&m_device_atom_positions, m_num_atoms * sizeof(AtomPosition));
    cudaMalloc(&m_device_reference_positions, m_num_atoms * sizeof(AtomPosition));
    cudaMalloc(&m_device_rotation_matrix, 3 * 3 * sizeof(double));
    cudaMalloc(&m_device_eigenvalues, 4 * sizeof(double));
    cudaMalloc(&m_device_eigenvectors, 4 * 4 * sizeof(double));
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&m_center_tmp, sizeof(AtomPosition));
    // initialize the buffer of CUDA eigen solver
    cusolverH = NULL;
    cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusolver_status = cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, m_stream);
    cusolverDnCreateSyevjInfo(&syevj_info);
    cusolverDnXsyevjSetTolerance(syevj_info, 1e-5);
    cusolverDnXsyevjSetMaxSweeps(syevj_info, 50);
    lwork = 0;
    jobz = CUSOLVER_EIG_MODE_VECTOR;
    uplo = CUBLAS_FILL_MODE_LOWER;
    const size_t n_cols = 4;
    cusolver_status = cusolverDnDsyevj_bufferSize(cusolverH, jobz, uplo, n_cols, m_device_eigenvectors, n_cols, m_device_eigenvalues, &lwork, syevj_info);
    device_work = nullptr;
    cudaMalloc(&device_work, lwork * sizeof(double));
    cudaMalloc(&m_device_rmsd, 1 * sizeof(double));
    cudaMallocHost(&m_host_rmsd, 1 * sizeof(double));
    cudaMalloc(&d_count, 1 * sizeof(unsigned int));
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
}

void OptimalRotation::updateReference(const std::vector<AtomPosition>& reference_positions) {
    cudaMemcpyAsync(m_device_reference_positions, reference_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream);
    bringToCenterDevice(m_device_reference_positions, m_num_atoms);
    // cudaStreamSynchronize(m_stream);
}

void OptimalRotation::updateAtoms(const std::vector<AtomPosition>& atom_positions) {
    cudaMemcpyAsync(m_device_atom_positions, atom_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream);
    bringToCenterDevice(m_device_atom_positions, m_num_atoms);
    // cudaStreamSynchronize(m_stream);
}

void OptimalRotation::bringToCenterDevice(AtomPosition* device_atom_positions, const size_t num_atoms) {
    // const int num_blocks = int(std::ceil(double(m_num_atoms) / block_size));
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
    cudaMemsetAsync(m_center_tmp, 0, sizeof(double3), m_stream);
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    get_center_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, num_atoms, d_count);
    move_atom_to_center_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, num_atoms);
}

void OptimalRotation::calculateOptimalRotationMatrix() {
    const size_t n_cols = 4;
#ifdef DEBUG
    const size_t n_rows = 4;
#endif
    // const int block_size = 32;
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
    // build matrix F
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    cudaMemsetAsync(m_device_eigenvectors, 0, 4 * 4 * sizeof(double), m_stream);
    build_matrix_F_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(m_device_atom_positions, m_device_reference_positions, m_device_eigenvectors, m_num_atoms, d_count);

    // device_matrix_F is the eigenvectors after solving
    cusolver_status = cusolverDnDsyevj(cusolverH, jobz, uplo, n_cols, m_device_eigenvectors, n_cols, m_device_eigenvalues, device_work, lwork, devInfo, syevj_info);
    // cudaStreamSynchronize(m_stream);
    // build the optimal rotation matrix
    build_rotation_matrix_kernel<<<1,1,0,m_stream>>>(m_device_eigenvectors, m_device_rotation_matrix);
    // cudaStreamSynchronize(m_stream);
}

// compute the optimal rmsd
double OptimalRotation::minimalRMSD() const {
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
    cudaMemsetAsync(m_device_rmsd, 0, 1 * sizeof(double), m_stream);
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    compute_optimal_rmsd_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(
        m_device_atom_positions,
        m_device_reference_positions,
        m_device_eigenvalues,
        m_device_rotation_matrix,
        m_device_rmsd, m_num_atoms, d_count);
    cudaMemcpyAsync(m_host_rmsd, m_device_rmsd, 1 * sizeof(double), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    return *(m_host_rmsd);
}

// compute the optimal rmsd with respect to a specified frame
double OptimalRotation::minimalRMSD(const std::vector<AtomPosition>& atom_positions) const {
    cudaMemsetAsync(m_device_rmsd, 0, 1 * sizeof(double), m_stream);
    AtomPosition* device_atom_positions;
    cudaMalloc(&device_atom_positions, m_num_atoms * sizeof(AtomPosition));
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
    // copy data to device
    cudaMemcpyAsync(device_atom_positions, atom_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream);
    // compute geometric center
    cudaMemsetAsync(m_center_tmp, 0, sizeof(double3), m_stream);
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    get_center_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, m_num_atoms, d_count);
    move_atom_to_center_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, m_num_atoms);
    // we assume the reference frame is already moved to its center of geometry
    // rotate the atoms
    rotate_atoms_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_device_rotation_matrix, m_num_atoms);
    // compute rmsd directly
    cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    compute_rmsd_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_device_reference_positions, m_device_rmsd, m_num_atoms, d_count);
    cudaMemcpyAsync(m_host_rmsd, m_device_rmsd, 1 * sizeof(double), cudaMemcpyDeviceToHost, m_stream);
//     cudaFree(device_rmsd);
    cudaStreamSynchronize(m_stream);
    cudaFree(device_atom_positions);
    return *(m_host_rmsd);
}

OptimalRotation::~OptimalRotation() {
    cudaFree(m_device_atom_positions);
    cudaFree(m_device_reference_positions);
    cudaFree(m_device_rotation_matrix);
    cudaFree(m_device_eigenvalues);
    cudaFree(m_device_eigenvectors);
    cudaStreamDestroy(m_stream);
    cusolverDnDestroySyevjInfo(syevj_info);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaFree(devInfo);
    cudaFree(m_center_tmp);
    cudaFree(device_work);
    cudaFree(m_device_rmsd);
    cudaFreeHost(m_host_rmsd);
    cudaFree(d_count);
    cudaDeviceReset();
}
