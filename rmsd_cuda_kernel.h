#ifndef RMSD_CUDA_KERNEL_H
#define RMSD_CUDA_KERNEL_H

#include "cub/cub.cuh"

// eq 33 in "Using Quaternions to Calculate RMSD"
__global__ void build_rotation_matrix_kernel(double* eigenvectors, double* rotation_matrix, const size_t max_eigenvalue_index = 3);

// move atoms to a given center
__global__ void move_atom_to_center_kernel(double3* atom_positions, double3* center, const size_t num_atoms);

// compute the center of geometry from a set of atoms
template <int block_size>
__global__ void get_center_kernel(double3* atom_positions, double3* center, const size_t num_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double center_x = 0, center_y = 0, center_z = 0;
    if (i < num_atoms) {
        center_x = atom_positions[i].x;
        center_y = atom_positions[i].y;
        center_z = atom_positions[i].z;
    }
    typedef cub::BlockReduce<double, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    center_x = BlockReduce(temp_storage).Sum(center_x);
    __syncthreads();
    center_y = BlockReduce(temp_storage).Sum(center_y);
    __syncthreads();
    center_z = BlockReduce(temp_storage).Sum(center_z);
    __syncthreads();
    if (threadIdx.x == 0) {
        center->x = center_x / double(num_atoms);
        center->y = center_y / double(num_atoms);
        center->z = center_z / double(num_atoms);
    }
}

// step 3, section "The Algorithm", page 1855, "Using Quaternions to Calculate RMSD"
template <int block_size>
__global__ void compute_optimal_rmsd_kernel(const double3* atom_positions, const double3* reference_positions, const double* eigenvalues, double* rmsd, const size_t num_atoms, const size_t max_eigenvalue_index = 3) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (i < num_atoms) {
        sum = atom_positions[i].x * atom_positions[i].x + 
              atom_positions[i].y * atom_positions[i].y +
              atom_positions[i].z * atom_positions[i].z +
              reference_positions[i].x * reference_positions[i].x +
              reference_positions[i].y * reference_positions[i].y +
              reference_positions[i].z * reference_positions[i].z;
    }
    typedef cub::BlockReduce<double, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    if (threadIdx.x == 0) {
        *(rmsd) = sqrt((sum - 2.0 * eigenvalues[max_eigenvalue_index]) / double(num_atoms));
    }
}

// directly compute RMSD
template <int block_size>
__global__ void compute_rmsd_kernel(const double3* atom_positions, const double3* reference_positions, double* rmsd, const size_t num_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (i < num_atoms) {
        const double xx = (atom_positions[i].x - reference_positions[i].x) * (atom_positions[i].x - reference_positions[i].x);
        const double yy = (atom_positions[i].y - reference_positions[i].y) * (atom_positions[i].y - reference_positions[i].y);
        const double zz = (atom_positions[i].z - reference_positions[i].z) * (atom_positions[i].z - reference_positions[i].z);
        sum = xx + yy + zz;
    }
    typedef cub::BlockReduce<double, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    if (threadIdx.x == 0) {
        *(rmsd) = sqrt(sum / double(num_atoms));
    }
}

// rotate atoms in place
template <int block_size>
__global__ void rotate_atoms_kernel(double3* atom_positions, const double* rotation_matrix, const size_t num_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_atoms) {
        const double atom_x = atom_positions[i].x;
        const double atom_y = atom_positions[i].y;
        const double atom_z = atom_positions[i].z;
        // R_ij = rotation_matrix[i * 3 + j]
        atom_positions[i].x = rotation_matrix[0 * 3 + 0] * atom_x + rotation_matrix[0 * 3 + 1] * atom_y + rotation_matrix[0 * 3 + 2] * atom_z;
        atom_positions[i].y = rotation_matrix[1 * 3 + 0] * atom_x + rotation_matrix[1 * 3 + 1] * atom_y + rotation_matrix[1 * 3 + 2] * atom_z;
        atom_positions[i].z = rotation_matrix[2 * 3 + 0] * atom_x + rotation_matrix[2 * 3 + 1] * atom_y + rotation_matrix[2 * 3 + 2] * atom_z;
    }
}

// eq 10, "Using Quaternions to Calculate RMSD"
template <int block_size>
__global__ void build_matrix_F_kernel(const double3* atom_positions, const double3* reference_positions, double* matrix_F, const size_t num_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double R11 = 0;
    double R22 = 0;
    double R33 = 0;
    double R12 = 0;
    double R13 = 0;
    double R23 = 0;
    double R21 = 0;
    double R31 = 0;
    double R32 = 0;
    if (i < num_atoms) {
        R11 = atom_positions[i].x * reference_positions[i].x;
        R22 = atom_positions[i].y * reference_positions[i].y;
        R33 = atom_positions[i].z * reference_positions[i].z;
        R12 = atom_positions[i].x * reference_positions[i].y;
        R13 = atom_positions[i].x * reference_positions[i].z;
        R23 = atom_positions[i].y * reference_positions[i].z;
        R21 = atom_positions[i].y * reference_positions[i].x;
        R31 = atom_positions[i].z * reference_positions[i].x;
        R32 = atom_positions[i].z * reference_positions[i].y;
    }
    typedef cub::BlockReduce<double, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    R11 = BlockReduce(temp_storage).Sum(R11);
    __syncthreads();
    R22 = BlockReduce(temp_storage).Sum(R22);
    __syncthreads();
    R33 = BlockReduce(temp_storage).Sum(R33);
    __syncthreads();
    R12 = BlockReduce(temp_storage).Sum(R12);
    __syncthreads();
    R13 = BlockReduce(temp_storage).Sum(R13);
    __syncthreads();
    R23 = BlockReduce(temp_storage).Sum(R23);
    __syncthreads();
    R21 = BlockReduce(temp_storage).Sum(R21);
    __syncthreads();
    R31 = BlockReduce(temp_storage).Sum(R31);
    __syncthreads();
    R32 = BlockReduce(temp_storage).Sum(R32);
    __syncthreads();
    if (threadIdx.x == 0) {
        const size_t n_cols = 4;
        matrix_F[0 + 0 * n_cols] = R11 + R22 + R33;
        matrix_F[1 + 0 * n_cols] = R23 - R32;
        matrix_F[2 + 0 * n_cols] = R31 - R13;
        matrix_F[3 + 0 * n_cols] = R12 - R21;
        matrix_F[0 + 1 * n_cols] = matrix_F[1 + 0 * n_cols];
        matrix_F[1 + 1 * n_cols] = R11 - R22 - R33;
        matrix_F[2 + 1 * n_cols] = R12 + R21;
        matrix_F[3 + 1 * n_cols] = R13 + R31;
        matrix_F[0 + 2 * n_cols] = matrix_F[2 + 0 * n_cols];
        matrix_F[1 + 2 * n_cols] = matrix_F[2 + 1 * n_cols];
        matrix_F[2 + 2 * n_cols] = -R11 + R22 - R33;;
        matrix_F[3 + 2 * n_cols] = R23 + R32;
        matrix_F[0 + 3 * n_cols] = matrix_F[3 + 0 * n_cols];
        matrix_F[1 + 3 * n_cols] = matrix_F[3 + 1 * n_cols];
        matrix_F[2 + 3 * n_cols] = matrix_F[3 + 2 * n_cols];
        matrix_F[3 + 3 * n_cols] = -R11 - R22 + R33;
    }
}

#endif
