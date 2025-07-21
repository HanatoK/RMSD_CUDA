#include "rmsd_cuda_kernel.cuh"

// eq 33 in "Using Quaternions to Calculate RMSD"
__global__ void build_rotation_matrix_kernel(double* eigenvectors, double* rotation_matrix, const size_t max_eigenvalue_index) {
    if (threadIdx.x == 0) {
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

template <typename T> __inline__ __device__ int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

__inline__ __device__ void apply_jacobi(
    // const double* __restrict old_A,
    double* __restrict A,
    int p, int q, double c, double s) {
    const double c2 = c*c;
    const double s2 = s*s;
    const double cs = c*s;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double oip = A[i*4+p];
        const double oiq = A[i*4+q];
        if (i != p && i != q) {
            A[i*4+p] = c * oip - s * oiq;
            A[p*4+i] = A[i*4+p];
            A[i*4+q] = c * oiq + s * oip;
            A[q*4+i] = A[i*4+q];
        }
    }
    const double opp = A[p*4+p];
    const double oqq = A[q*4+q];
    const double opq = A[p*4+q];
    A[p*4+p] = c2 * opp + s2 * oqq - 2.0 * cs * opq;
    A[q*4+q] = s2 * opp + c2 * oqq + 2.0 * cs * opq;
    A[p*4+q] = 0;
    A[q*4+p] = 0;
}

__inline__ __device__ void multiply_jacobi(
    double* __restrict V, int p, int q, double c, double s) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double oip = V[i*4+p];
        const double oiq = V[i*4+q];
        V[i*4+p] = c * oip - s * oiq;
        V[i*4+q] = s * oip + c * oiq;
    }
}

// Use exactly 16 threads
__global__ void jacobi_4x4(double* A_in, double* eigvals, int* max_reached) {
    __shared__ double A[4*4], V[4*4];
    const int idx = threadIdx.x;
    const int i = idx / 4;
    const int j = idx % 4;
    if (max_reached && idx == 0) {
        max_reached[0] = 0;
        __threadfence();
    }
    V[idx] = double(i == j);
    A[idx] = A_in[idx];
    __syncthreads();
    // printf("(in) idx = %d, A[%d] = %12.7f\n", idx, idx, A[idx]);
    const int max_iteration = 50;
    double off_diag_sum = (j > i) ? A[idx] * A[idx] : 0.0;
    // printf("idx = %d, off_diag = %f\n", off_diag_sum);
    typedef cub::WarpReduce<double, 16> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;
    off_diag_sum = WarpReduce(temp_storage).Sum(off_diag_sum);
    __syncwarp();
    off_diag_sum = __shfl_sync(0xFFFFFFFF, off_diag_sum, 0);
    __syncwarp();
    int iteration = 0;
    while (off_diag_sum > 1e-16) {
        double c, s;
        // Apply Jacobi rotation
        #pragma unroll
        for (int i0 = 0; i0 < 4; ++i0) {
            for (int j0 = i0 + 1; j0 < 4; ++j0) {
                if (idx == 0) {
                    const double a_pq = A[i0*4+j0];
                    const double a_pp = A[i0*4+i0];
                    const double a_qq = A[j0*4+j0];
                    const double theta = 0.5 * (a_qq - a_pp) / a_pq;
                    const double sign = sgn(theta) == 0 ? 1.0 : sgn(theta);
                    const double t = sign / (abs(theta) + sqrt(theta * theta + 1.0));
                    c = 1.0 / sqrt(t * t + 1.0);
                    s = t * c;
                    apply_jacobi(A, i0, j0, c, s);
                    multiply_jacobi(V, i0, j0, c, s);
                }
            }
        }
        __syncthreads();
        // Compute off-diagonal sum
        off_diag_sum = (j > i) ? A[idx] * A[idx] : 0.0;
        off_diag_sum = WarpReduce(temp_storage).Sum(off_diag_sum);
        __syncwarp();
        off_diag_sum = __shfl_sync(0xFFFFFFFF, off_diag_sum, 0);
        __syncwarp();
        // Check the number of iterations
        ++iteration;
        if (iteration > max_iteration) {
            if (idx == 0 && max_reached) atomicAdd(max_reached, 1);
            break;
        }
    }
    __syncthreads();
    // Sort
    double p;
    if (idx == 0) {
        int k;
        #pragma unroll
        for (int i0 = 0; i0 < 4; ++i0) {
            k = i0;
            p = A[i0*4+i0];
            for (int j0 = i0 + 1; j0 < 4; ++j0) {
                if (A[j0*4+j0] <= p) {
                    k = j0;
                    p = A[j0*4+j0];
                }
            }
            if (k != i0) {
                A[k*4+k] = A[i0*4+i0];
                A[i0*4+i0] = p;
                for (int j0 = 0; j0 < 4; ++j0) {
                    p = V[j0*4+i0];
                    V[j0*4+i0] = V[j0*4+k];
                    V[j0*4+k] = p;
                }
            }
        }
    }
    __syncthreads();
    // Transpose
    A_in[i*4+j] = V[j*4+i];
    if (i == j) eigvals[i] = A[idx];
}
