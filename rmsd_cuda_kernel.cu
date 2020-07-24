#include "rmsd_cuda_kernel.h"

// eq 33 in "Using Quaternions to Calculate RMSD"
__global__ void build_rotation_matrix_kernel(double* eigenvectors, double* rotation_matrix, const size_t max_eigenvalue_index) {
    double q[4];
    q[0] = eigenvectors[max_eigenvalue_index * 4 + 0];
    q[1] = eigenvectors[max_eigenvalue_index * 4 + 1];
    q[2] = eigenvectors[max_eigenvalue_index * 4 + 2];
    q[3] = eigenvectors[max_eigenvalue_index * 4 + 3];
    rotation_matrix[0 * 3 + 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
    rotation_matrix[0 * 3 + 1] = 2.0 * (q[1] * q[2] - q[0] * q[3]);
    rotation_matrix[0 * 3 + 2] = 2.0 * (q[1] * q[3] + q[0] * q[2]);
    rotation_matrix[1 * 3 + 0] = 2.0 * (q[1] * q[2] + q[0] * q[3]);
    rotation_matrix[1 * 3 + 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
    rotation_matrix[1 * 3 + 2] = 2.0 * (q[2] * q[3] - q[0] * q[1]);
    rotation_matrix[2 * 3 + 0] = 2.0 * (q[1] * q[3] - q[0] * q[2]);
    rotation_matrix[2 * 3 + 1] = 2.0 * (q[2] * q[3] + q[0] * q[1]);
    rotation_matrix[2 * 3 + 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
}

// move atoms to a given center
__global__ void move_atom_to_center_kernel(double3* atom_positions, double3* center, const size_t num_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_atoms) {
        atom_positions[i].x -= center->x;
        atom_positions[i].y -= center->y;
        atom_positions[i].z -= center->z;
    }
}
