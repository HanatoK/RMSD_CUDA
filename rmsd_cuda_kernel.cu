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

template <int p, int q>
__inline__ __device__ void apply_jacobi(
    // const double* __restrict old_A,
    double* __restrict A,
    const double c,
    const double s,
    const double c2,
    const double s2,
    const double cs) {
    // const double c2 = c*c;
    // const double s2 = s*s;
    // const double cs = c*s;
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

template <int p, int q>
__inline__ __device__ void multiply_jacobi(
    double* __restrict V, double c, double s) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double oip = V[i*4+p];
        const double oiq = V[i*4+q];
        V[i*4+p] = c * oip - s * oiq;
        V[i*4+q] = s * oip + c * oiq;
    }
}

__inline__ __device__ void compute_c_s(
    double a_pq, double a_pp, double a_qq, double& c, double& s, double& c2, double& s2, double& cs) {
    const double theta = 0.5 * (a_qq - a_pp) / a_pq;
    const double t = 1 / (sqrt(theta * theta + 1.0) + fabs(theta));
    // const double t = sqrt(theta * theta + 1.0) - fabs(theta);
    c = rsqrt(t * t + 1.0);
    s = theta < 0 ? -t * c : t * c;
    c2 = c*c;
    s2 = s*s;
    cs = c*s;
    // const double phi = 0.5 * atan2(2 * a_pq, a_qq - a_pp);
    // sincos(phi, &s, &c);
}

// Use exactly 2 threads
__global__ void jacobi_4x4(double* A_in, double* eigvals, int* max_reached) {
    __shared__ double A[4*4];
    __shared__ double V[4*4];
    const int idx = threadIdx.x;
    // const int i = idx / 4;
    // const int j = idx % 4;
    if (max_reached && idx == 0) {
        max_reached[0] = 0;
        __threadfence();
    }
    if (idx == 0) {
        // V[idx] = double(i == j);
        // A[idx] = A_in[idx];
        memset(V, 0, sizeof(double)*4*4);
        // __threadfence();
        V[0*4+0] = 1;
        V[1*4+1] = 1;
        V[2*4+2] = 1;
        V[3*4+3] = 1;
        // memcpy(A, A_in, 4*4*sizeof(double));
        A[0] = A_in[0];
        A[1] = A_in[1];
        A[2] = A_in[2];
        A[3] = A_in[3];
        A[4] = A_in[4];
        A[5] = A_in[5];
        A[6] = A_in[6];
        A[7] = A_in[7];
        A[8] = A_in[8];
        A[9] = A_in[9];
        A[10] = A_in[10];
        A[11] = A_in[11];
        A[12] = A_in[12];
        A[13] = A_in[13];
        A[14] = A_in[14];
        A[15] = A_in[15];
        // __threadfence();
    }
    __syncthreads();
    const int p_ids[] = {0, 2, 0, 1, 0, 1};
    const int q_ids[] = {1, 3, 2, 3, 3, 2};
    constexpr int max_iteration = 50;
    double off_diag_sum =
        A[0*4+1]*A[0*4+1]+A[0*4+2]*A[0*4+2]+A[0*4+3]*A[0*4+3]+
        A[1*4+2]*A[1*4+2]+A[1*4+3]*A[1*4+3]+
        A[2*4+3]*A[2*4+3];
    // off_diag_sum = __shfl_sync(0xFFFFFFFF, off_diag_sum, 0);
    // __syncwarp();
    int iteration = 0;
    while (off_diag_sum > 1e-16) {
        double c = 0, s = 0, c2 = 0, s2 = 0, cs = 0;
        bool rotate = false;
        int p = p_ids[idx];
        int q = q_ids[idx];
        double a_pq = A[p*4+q];
        if (fabs(a_pq) > 0) {
            rotate = true;
            const double a_pp = A[p*4+p];
            const double a_qq = A[q*4+q];
            compute_c_s(a_pq, a_pp, a_qq, c, s, c2, s2, cs);
        }
        __syncwarp();
        if (idx == 0 && rotate) {
            apply_jacobi<0, 1>(A, c, s, c2, s2, cs);
            multiply_jacobi<0, 1>(V, c, s);
        }
        __syncwarp();
        if (idx == 1 && rotate) {
            apply_jacobi<2, 3>(A, c, s, c2, s2, cs);
            multiply_jacobi<2, 3>(V, c, s);
        }
        __syncwarp();
        rotate = false;
        p = p_ids[idx+2];
        q = q_ids[idx+2];
        a_pq = A[p*4+q];
        if (fabs(a_pq) > 0) {
            rotate = true;
            const double a_pp = A[p*4+p];
            const double a_qq = A[q*4+q];
            compute_c_s(a_pq, a_pp, a_qq, c, s, c2, s2, cs);
        }
        __syncwarp();
        if (idx == 0 && rotate) {
            apply_jacobi<0, 2>(A, c, s, c2, s2, cs);
            multiply_jacobi<0, 2>(V, c, s);
        }
        __syncwarp();
        if (idx == 1 && rotate) {
            apply_jacobi<1, 3>(A, c, s, c2, s2, cs);
            multiply_jacobi<1, 3>(V, c, s);
        }
        __syncwarp();
        rotate = false;
        p = p_ids[idx+4];
        q = q_ids[idx+4];
        a_pq = A[p*4+q];
        if (fabs(a_pq) > 0) {
            rotate = true;
            const double a_pp = A[p*4+p];
            const double a_qq = A[q*4+q];
            compute_c_s(a_pq, a_pp, a_qq, c, s, c2, s2, cs);
        }
        __syncwarp();
        if (idx == 0 && rotate) {
            apply_jacobi<0, 3>(A, c, s, c2, s2, cs);
            multiply_jacobi<0, 3>(V, c, s);
        }
        __syncwarp();
        if (idx == 1 && rotate) {
            apply_jacobi<1, 2>(A, c, s, c2, s2, cs);
            multiply_jacobi<1, 2>(V, c, s);
        }
        __syncwarp();
        off_diag_sum =
            A[0*4+1]*A[0*4+1]+A[0*4+2]*A[0*4+2]+A[0*4+3]*A[0*4+3]+
            A[1*4+2]*A[1*4+2]+A[1*4+3]*A[1*4+3]+
            A[2*4+3]*A[2*4+3];
        // Check the number of iterations
        ++iteration;
        if (iteration > max_iteration) {
            if (idx == 0 && max_reached) atomicAdd(max_reached, 1);
            break;
        }
    }
    // __syncthreads();
    // Sort
    double p;
    if (idx == 0) {
        int k;
        // #pragma unroll
        // for (int i0 = 0; i0 < 4; ++i0) {
        //     k = i0;
        //     p = A[i0*4+i0];
        //     for (int j0 = i0 + 1; j0 < 4; ++j0) {
        //         if (A[j0*4+j0] < p) {
        //             k = j0;
        //             p = A[j0*4+j0];
        //         }
        //     }
        //     if (k != i0) {
        //         A[k*4+k] = A[i0*4+i0];
        //         A[i0*4+i0] = p;
        //         for (int j0 = 0; j0 < 4; ++j0) {
        //             p = V[j0*4+i0];
        //             V[j0*4+i0] = V[j0*4+k];
        //             V[j0*4+k] = p;
        //         }
        //     }
        // }
        k = 0;
        p = A[0*4+0];
        // Find the index of the leading eigenvalue
        #pragma unroll
        for (int i0 = 1; i0 < 4; ++i0) {
            if (A[i0*4+i0] > p) {
                p = A[i0*4+i0];
                k = i0;
            }
        }
        if (k != 0) {
            // Move the leading eigenvalue
            A[k*4+k] = A[0*4+0];
            A[0*4+0] = p;
            #pragma unroll
            for (int j0 = 0; j0 < 4; ++j0) {
                p = V[j0*4+0];
                V[j0*4+0] = V[j0*4+k];
                V[j0*4+k] = p;
            }
        }
    }
    // __syncthreads();
    // Transpose
    // A_in[i*4+j] = V[j*4+i];
    // if (i == j) eigvals[i] = A[idx];
    if (idx == 0) {
        A_in[0] = V[0];
        A_in[1] = V[4];
        A_in[2] = V[8];
        A_in[3] = V[12];
        A_in[4] = V[1];
        A_in[5] = V[5];
        A_in[6] = V[9];
        A_in[7] = V[13];
        A_in[8] = V[2];
        A_in[9] = V[6];
        A_in[10] = V[10];
        A_in[11] = V[14];
        A_in[12] = V[3];
        A_in[13] = V[7];
        A_in[14] = V[11];
        A_in[15] = V[15];
        eigvals[0] = A[0*4+0];
        eigvals[1] = A[1*4+1];
        eigvals[2] = A[2*4+2];
        eigvals[3] = A[3*4+3];
    }
}
