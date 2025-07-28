#include "rmsd_cuda.h"
#include "rmsd_cuda_kernel.cuh"
#include <cusolverDn.h>
#include <iostream>

template <typename T>
void add_copy_node(
  const T* src, T* dst, size_t num_elements,
  cudaMemcpyKind kind, cudaGraphNode_t& node_out, cudaGraph_t& graph,
  std::vector<cudaGraphNode_t> dependencies) {
  cudaMemcpy3DParms    memcpyParams     = {0};
  memcpyParams.kind     = kind;
  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr   = make_cudaPitchedPtr(
    (void*)src, sizeof(T) * num_elements, num_elements, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr   = make_cudaPitchedPtr(
    (void*)dst, sizeof(T) * num_elements, num_elements, 1);
  memcpyParams.extent   = make_cudaExtent(sizeof(T) * num_elements, 1, 1);
  checkCudaErrors(
    cudaGraphAddMemcpyNode(
      &node_out, graph, dependencies.data(),
      dependencies.size(), &memcpyParams));
}

template <typename T>
void update_copy_node(
  const T* src, T* dst, size_t num_elements,
  cudaMemcpyKind kind, cudaGraphNode_t& node,
  cudaGraphExec_t& graph_exec) {
  cudaMemcpy3DParms    memcpyParams     = {0};
  memcpyParams.kind     = kind;
  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr   = make_cudaPitchedPtr(
    (void*)src, sizeof(T) * num_elements, num_elements, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr   = make_cudaPitchedPtr(
    (void*)dst, sizeof(T) * num_elements, num_elements, 1);
  memcpyParams.extent   = make_cudaExtent(sizeof(T) * num_elements, 1, 1);
  checkCudaErrors(cudaGraphExecMemcpyNodeSetParams(
    graph_exec, node,
    &memcpyParams));
}

template <typename T>
void add_clear_array_node(
  T* dst, const size_t num_elements,
  cudaGraphNode_t& node_out, cudaGraph_t& graph,
  const std::vector<cudaGraphNode_t>& dependencies) {
  // size_t elementSize, width;
  const size_t sizeofT = sizeof(T);
  /**< Size of each element in bytes. Must be 1, 2, or 4. */
  const size_t elementSize =
    (sizeofT % 4 == 0) ? 4 :
    ((sizeofT % 2 == 0) ? 2 : 1);
  const size_t width = num_elements * (sizeofT / elementSize);
  cudaMemsetParams memsetParams = {0};
  memsetParams.dst         = (void*)dst;
  memsetParams.value       = 0;
  memsetParams.elementSize = elementSize;
  memsetParams.width       = width;
  memsetParams.height      = 1;
  checkCudaErrors(cudaGraphAddMemsetNode(
    &node_out, graph, dependencies.data(),
    dependencies.size(), &memsetParams));
}

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

OptimalRotation::OptimalRotation(const size_t num_atoms)
#if defined (USE_CUDA_GRAPH)
    : graphCreated(false)
#endif
{
    checkCudaErrors(cudaStreamCreate(&m_stream));
#if defined(USE_CUDA_GRAPH)
    std::memset(&gpu_nodes, 0, sizeof(gpu_nodes));
    checkCudaErrors(cudaGraphCreate(&m_graph, 0));
    m_instance = NULL;
#endif
    m_num_atoms = num_atoms;
    std::cout << "Number of atoms: " << num_atoms << std::endl;
    checkCudaErrors(cudaMalloc(&m_device_atom_positions, m_num_atoms * sizeof(AtomPosition)));
    checkCudaErrors(cudaMalloc(&m_device_reference_positions, m_num_atoms * sizeof(AtomPosition)));
    checkCudaErrors(cudaMalloc(&m_device_rotation_matrix, 3 * 3 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&m_device_eigenvalues, 4 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&m_device_eigenvectors, 4 * 4 * sizeof(double)));
#if defined (USE_CUDA_GRAPH)
    checkCudaErrors(cudaMalloc(&m_center_tmp_ref, sizeof(AtomPosition)));
    checkCudaErrors(cudaMalloc(&m_center_tmp_pos, sizeof(AtomPosition)));
#else
    checkCudaErrors(cudaMalloc(&m_center_tmp, sizeof(AtomPosition)));
#endif // USE_CUDA_GRAPH
    checkCudaErrors(cudaMalloc(&m_device_rmsd, 1 * sizeof(double)));
#if !defined (USE_NR)
    // initialize the buffer of CUDA eigen solver
    cusolverH = NULL;
    cudaMalloc(&devInfo, sizeof(int));
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
#endif // !defined (USE_NR)
// #if defined(USE_CUDA_GRAPH)
    checkCudaErrors(cudaMallocHost(&m_host_rmsd, 1 * sizeof(double)));
#if defined (USE_CUDA_GRAPH)
    checkCudaErrors(cudaMalloc(&d_count_ref, 1 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_count_pos, 1 * sizeof(unsigned int)));
#endif
    checkCudaErrors(cudaMalloc(&d_count, 1 * sizeof(unsigned int)));
    // cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream);
    mEventAttrib.version = NVTX_VERSION;
    mEventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    mEventAttrib.colorType = NVTX_COLOR_ARGB;
    mEventAttrib.color = 0xFF880000;
    mEventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
#if defined (USE_NR)
    mEventAttrib.message.ascii = "NR";
#else
    mEventAttrib.message.ascii = "cuSolver";
#endif
#if defined (USE_NR)
    checkCudaErrors(cudaMallocHost(&max_reached, sizeof(int)));
    max_reached[0] = 0;
#endif
}

void OptimalRotation::updateReference(const host_vector<AtomPosition>& reference_positions) {
#if defined (USE_CUDA_GRAPH)
    if (graphCreated == false) {
        add_copy_node(
            reference_positions.data(),
            m_device_reference_positions,
            m_num_atoms, cudaMemcpyHostToDevice,
            gpu_nodes.updateReferenceNode, m_graph, {});
        bringToCenterDevice(
            m_device_reference_positions, m_num_atoms,
            m_center_tmp_ref, d_count_ref,
            {gpu_nodes.updateReferenceNode},
            gpu_nodes.centerReferenceNode);
    } else {
        update_copy_node(
            reference_positions.data(),
            m_device_reference_positions,
            m_num_atoms, cudaMemcpyHostToDevice,
            gpu_nodes.updateReferenceNode, m_instance);
    }
#else
    checkCudaErrors(cudaMemcpyAsync(m_device_reference_positions, reference_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream));
    bringToCenterDevice(m_device_reference_positions, m_num_atoms);
    // cudaStreamSynchronize(m_stream);
#endif
}

void OptimalRotation::updateAtoms(const host_vector<AtomPosition>& atom_positions) {
#if defined (USE_CUDA_GRAPH)
    if (graphCreated == false) {
        add_copy_node(
            atom_positions.data(),
            m_device_atom_positions,
            m_num_atoms, cudaMemcpyHostToDevice,
            gpu_nodes.updateAtomsNode, m_graph, {});
        bringToCenterDevice(
            m_device_atom_positions, m_num_atoms,
            m_center_tmp_pos, d_count_pos,
            {gpu_nodes.updateAtomsNode},
            gpu_nodes.centerAtomsNode);
    } else {
        update_copy_node(
            atom_positions.data(),
            m_device_atom_positions,
            m_num_atoms, cudaMemcpyHostToDevice,
            gpu_nodes.updateAtomsNode, m_instance);
    }
#else
    checkCudaErrors(cudaMemcpyAsync(m_device_atom_positions, atom_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream));
    bringToCenterDevice(m_device_atom_positions, m_num_atoms);
    // cudaStreamSynchronize(m_stream);
#endif
}

void OptimalRotation::bringToCenterDevice(AtomPosition* device_atom_positions, const size_t num_atoms
#if defined(USE_CUDA_GRAPH)
, AtomPosition* center_out,
unsigned int* counter,
std::vector<cudaGraphNode_t> dependencies,
cudaGraphNode_t& last_node
#endif
) {
    // const int num_blocks = int(std::ceil(double(m_num_atoms) / block_size));
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
#if defined(USE_CUDA_GRAPH)
    if (graphCreated == false) {
        cudaGraphNode_t counterSetNode;
        cudaGraphNode_t centerSetNode;
        cudaGraphNode_t getCenterKernelNode;
        cudaGraphNode_t moveToCenterKernelNode;
        // cudaMemsetParams memsetParams = {0};
        // memsetParams.dst            = d_count;
        // memsetParams.value          = 0;
        // memsetParams.elementSize    = 1 * sizeof(unsigned int);
        // memsetParams.width          = 1;
        // memsetParams.height         = 1;
        // checkCudaErrors(cudaGraphAddMemsetNode(&counterSetNode, m_graph, dependencies.data(), dependencies.size(), &memsetParams));
        // memsetParams.dst            = m_center_tmp;
        // memsetParams.elementSize    = sizeof(float);
        // memsetParams.width          = 1 * sizeof(AtomPosition) / memsetParams.elementSize;
        // checkCudaErrors(cudaGraphAddMemsetNode(&centerSetNode, m_graph, dependencies.data(), dependencies.size(), &memsetParams));
        add_clear_array_node(counter, 1, counterSetNode, m_graph, {});
        add_clear_array_node(center_out, 1, centerSetNode, m_graph, {});
        // Run kernels
        cudaKernelNodeParams kernelNodeParams = {0};
        // cudaGraphNode_t dependencies[] = {
        //     counterSetNode,
        //     centerSetNode};
        dependencies.push_back(counterSetNode);
        dependencies.push_back(centerSetNode);
        const void *getCenterKernelArgs[] =
            {&device_atom_positions, &center_out, &num_atoms, &counter};
        kernelNodeParams.func           = (void*)get_center_kernel<block_size>;
        kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);
        kernelNodeParams.blockDim       = dim3(block_size, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams   = const_cast<void**>(getCenterKernelArgs);
        kernelNodeParams.extra          = NULL;
        checkCudaErrors(cudaGraphAddKernelNode(&getCenterKernelNode, m_graph, dependencies.data(), dependencies.size(), &kernelNodeParams));
        // last_node = getCenterKernelNode;
        const void* moveAtomToCenterKernelArgs[] =
            {&device_atom_positions, &center_out, &num_atoms};
        kernelNodeParams.func           = (void*)move_atom_to_center_kernel;
        kernelNodeParams.kernelParams   = const_cast<void**>(moveAtomToCenterKernelArgs);
        checkCudaErrors(cudaGraphAddKernelNode(&moveToCenterKernelNode, m_graph, &getCenterKernelNode, 1, &kernelNodeParams));
        last_node = moveToCenterKernelNode;
    }
#else
    checkCudaErrors(cudaMemsetAsync(m_center_tmp, 0, sizeof(double3), m_stream));
    checkCudaErrors(cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream));
    get_center_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, num_atoms, d_count);
    move_atom_to_center_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, num_atoms);
#endif
}

void OptimalRotation::calculateOptimalRotationMatrix() {
#ifdef DEBUG
    const size_t n_rows = 4;
#endif
    // const int block_size = 32;
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
#if defined(USE_CUDA_GRAPH)
    if (graphCreated == false) {
        // Memsets
        cudaGraphNode_t counterSetNode, eigenVectorsSetNode;
        add_clear_array_node(
            d_count, 1, counterSetNode, m_graph,
            {});
        add_clear_array_node(
            m_device_eigenvectors, 4*4,
            eigenVectorsSetNode, m_graph,
            {});
        // last_node = eigenVectorsSetNode;
        // build matrix F
        std::vector<cudaGraphNode_t> dependencies =
            {counterSetNode,
             eigenVectorsSetNode,
            gpu_nodes.centerReferenceNode,
            gpu_nodes.centerAtomsNode
            };
        cudaKernelNodeParams kernelNodeParams = {0};
        const void *buildMatrixFKernelArgs[] =
            {&m_device_atom_positions, &m_device_reference_positions, &m_device_eigenvectors, &m_num_atoms, &d_count};
        kernelNodeParams.func           = (void*)build_matrix_F_kernel<block_size>;
        kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);
        kernelNodeParams.blockDim       = dim3(block_size, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams   =
            const_cast<void**>(buildMatrixFKernelArgs);
        kernelNodeParams.extra          = NULL;
        cudaGraphAddKernelNode(&gpu_nodes.buildMatrixFNode, m_graph, dependencies.data(), dependencies.size(), &kernelNodeParams);
        // last_node = buildMatrixFNode;
    }
#else
    // build matrix F
    checkCudaErrors(cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream));
    checkCudaErrors(cudaMemsetAsync(m_device_eigenvectors, 0, 4 * 4 * sizeof(double), m_stream));
    build_matrix_F_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(m_device_atom_positions, m_device_reference_positions, m_device_eigenvectors, m_num_atoms, d_count);

    nvtxRangePushEx(&mEventAttrib);
#endif // defined(USE_CUDA_GRAPH)
#if defined(USE_NR)
#if defined(USE_CUDA_GRAPH)
    if (graphCreated == false) {
        // Jacobi node
        cudaKernelNodeParams kernelNodeParams = {0};
        const void *kernelArgs[] = {
            &m_device_eigenvectors, &m_device_eigenvalues, &max_reached};
        kernelNodeParams.func           = (void*)jacobi_4x4;
        kernelNodeParams.gridDim        = dim3(1, 1, 1);
        kernelNodeParams.blockDim       = dim3(2, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams   =
            const_cast<void**>(kernelArgs);
        kernelNodeParams.extra          = NULL;
        checkCudaErrors(cudaGraphAddKernelNode(&gpu_nodes.jacobi4x4Node, m_graph, &gpu_nodes.buildMatrixFNode, 1, &kernelNodeParams));
    }
    // getLastCudaError("Line 312\n");
#else
    jacobi_4x4<<<1,2,0,m_stream>>>(m_device_eigenvectors, m_device_eigenvalues, max_reached);
    if (max_reached[0] > 0) {
        std::cerr << "Maximum number of iterations reached!\n";
    }
#endif // defined(USE_CUDA_GRAPH)
#else
    const size_t n_cols = 4;
    cusolver_status = cusolverDnDsyevj(cusolverH, jobz, uplo, n_cols, m_device_eigenvectors, n_cols, m_device_eigenvalues, device_work, lwork, devInfo, syevj_info);
#endif
#if defined (USE_CUDA_GRAPH)
    if (graphCreated == false) {
        // Build rotation matrix node
        cudaKernelNodeParams kernelNodeParams = {0};
        size_t max_eigenvalue_index = 3;
        void *kernelArgs[] = {
            &m_device_eigenvectors, &m_device_rotation_matrix, &max_eigenvalue_index};
        kernelNodeParams.func           = (void*)build_rotation_matrix_kernel;
        kernelNodeParams.gridDim        = dim3(1, 1, 1);
        kernelNodeParams.blockDim       = dim3(1, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams   = kernelArgs;
        kernelNodeParams.extra          = NULL;
        checkCudaErrors(cudaGraphAddKernelNode(&gpu_nodes.buildRotationMatrixKernelNode, m_graph, &gpu_nodes.jacobi4x4Node, 1, &kernelNodeParams));
    }
#else
    nvtxRangePop();
    // cudaStreamSynchronize(m_stream);
    // build the optimal rotation matrix
    build_rotation_matrix_kernel<<<1,1,0,m_stream>>>(m_device_eigenvectors, m_device_rotation_matrix);
    // cudaStreamSynchronize(m_stream);
#endif
}

// compute the optimal rmsd
#if defined (USE_CUDA_GRAPH)
double OptimalRotation::minimalRMSD() {
#else
double OptimalRotation::minimalRMSD() const {
#endif
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
#if defined (USE_CUDA_GRAPH)
    if (graphCreated == false) {
        cudaGraphNode_t last_node = gpu_nodes.buildRotationMatrixKernelNode;
        // Memsets
        cudaGraphNode_t counterSetNode, deviceRMSDSetNode;
        cudaMemsetParams memsetParams = {0};
        memsetParams.dst            = d_count;
        memsetParams.value          = 0;
        memsetParams.elementSize    = 1 * sizeof(unsigned int);
        memsetParams.width          = 1;
        memsetParams.height         = 1;
        checkCudaErrors(cudaGraphAddMemsetNode(
            &counterSetNode, m_graph, &last_node, 1, &memsetParams));
        // last_node = counterSetNode;
        memsetParams.dst            = m_device_rmsd;
        memsetParams.elementSize    = sizeof(float);
        memsetParams.width          = sizeof(double) / memsetParams.elementSize;
        checkCudaErrors(cudaGraphAddMemsetNode(
            &deviceRMSDSetNode, m_graph, &last_node, 1, &memsetParams));
        // last_node = deviceRMSDSetNode;
        cudaGraphNode_t RMSDDependencies[] = {counterSetNode, deviceRMSDSetNode};
        // Compute RMSD
        cudaGraphNode_t RMSDKernelNode;
        cudaKernelNodeParams kernelNodeParams = {0};
        const size_t max_eigenvalue_index = 3;
        const void *kernelArgs[] = {
            &m_device_atom_positions,
            &m_device_reference_positions,
            &m_device_eigenvalues,
            &m_device_rotation_matrix,
            &m_device_rmsd, &m_num_atoms, &d_count,
            &max_eigenvalue_index};
        kernelNodeParams.func           =
            (void*)compute_optimal_rmsd_kernel<block_size>;
        kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);
        kernelNodeParams.blockDim       = dim3(block_size, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams   = const_cast<void**>(kernelArgs);
        kernelNodeParams.extra          = NULL;
        checkCudaErrors(cudaGraphAddKernelNode(
            &RMSDKernelNode, m_graph, RMSDDependencies, 2, &kernelNodeParams));
        // last_node = RMSDKernelNode;
        // Instantiate graph
        checkCudaErrors(cudaGraphInstantiate(&m_instance, m_graph, NULL, NULL, 0));
        graphCreated = true;
    }
    static int iter = 0;
    if (iter /*% 2*/ == 0) {
        cudaGraphDebugDotFlags dotFlags = cudaGraphDebugDotFlagsVerbose;
        const std::string filename = "graph_" + std::to_string(iter) + ".dot";
        checkCudaErrors(cudaGraphDebugDotPrint(m_graph, filename.c_str(), dotFlags));
    }
    iter++;
    // Run graph
    checkCudaErrors(cudaGraphLaunch(m_instance, m_stream));
#else
    checkCudaErrors(cudaMemsetAsync(m_device_rmsd, 0, 1 * sizeof(double), m_stream));
    checkCudaErrors(cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream));
    compute_optimal_rmsd_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(
        m_device_atom_positions,
        m_device_reference_positions,
        m_device_eigenvalues,
        m_device_rotation_matrix,
        m_device_rmsd, m_num_atoms, d_count);
#endif
    checkCudaErrors(cudaMemcpyAsync(m_host_rmsd, m_device_rmsd, 1 * sizeof(double), cudaMemcpyDeviceToHost, m_stream));
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    return *(m_host_rmsd);
}

#if defined (USE_CUDA_GRAPH)
void OptimalRotation::resetGraph() {
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    checkCudaErrors(cudaGraphExecDestroy(m_instance));
    checkCudaErrors(cudaGraphDestroy(m_graph));
    checkCudaErrors(cudaStreamDestroy(m_stream));
    graphCreated = false;
    // last_node = NULL;
    // Recreate graph
    checkCudaErrors(cudaStreamCreate(&m_stream));
    checkCudaErrors(cudaGraphCreate(&m_graph, 0));
}
#endif // USE_CUDA_GRAPH

#if defined (USE_CUDA_GRAPH)
#else
// compute the optimal rmsd with respect to a specified frame
double OptimalRotation::minimalRMSD(const host_vector<AtomPosition>& atom_positions) const {
    checkCudaErrors(cudaMemsetAsync(m_device_rmsd, 0, 1 * sizeof(double), m_stream));
    AtomPosition* device_atom_positions;
    checkCudaErrors(cudaMalloc(&device_atom_positions, m_num_atoms * sizeof(AtomPosition)));
    const int num_blocks = (m_num_atoms + block_size - 1) / block_size;
    // copy data to device
    checkCudaErrors(cudaMemcpyAsync(device_atom_positions, atom_positions.data(), m_num_atoms * sizeof(AtomPosition), cudaMemcpyHostToDevice, m_stream));
    // compute geometric center
    checkCudaErrors(cudaMemsetAsync(m_center_tmp, 0, sizeof(double3), m_stream));
    checkCudaErrors(cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream));
    get_center_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, m_num_atoms, d_count);
    move_atom_to_center_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_center_tmp, m_num_atoms);
    // we assume the reference frame is already moved to its center of geometry
    // rotate the atoms
    rotate_atoms_kernel<<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_device_rotation_matrix, m_num_atoms);
    // compute rmsd directly
    checkCudaErrors(cudaMemsetAsync(d_count, 0, 1 * sizeof(unsigned int), m_stream));
    compute_rmsd_kernel<block_size><<<num_blocks, block_size, 0, m_stream>>>(device_atom_positions, m_device_reference_positions, m_device_rmsd, m_num_atoms, d_count);
    checkCudaErrors(cudaMemcpyAsync(m_host_rmsd, m_device_rmsd, 1 * sizeof(double), cudaMemcpyDeviceToHost, m_stream));
//     cudaFree(device_rmsd);
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    checkCudaErrors(cudaFree(device_atom_positions));
    return *(m_host_rmsd);
}
#endif

OptimalRotation::~OptimalRotation() {
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    checkCudaErrors(cudaFree(m_device_atom_positions));
    checkCudaErrors(cudaFree(m_device_reference_positions));
    checkCudaErrors(cudaFree(m_device_rotation_matrix));
    checkCudaErrors(cudaFree(m_device_eigenvalues));
    checkCudaErrors(cudaFree(m_device_eigenvectors));
    checkCudaErrors(cudaStreamDestroy(m_stream));
#if !defined (USE_NR)
    cusolverDnDestroySyevjInfo(syevj_info);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    checkCudaErrors(cudaFree(devInfo));
    checkCudaErrors(cudaFree(device_work));
#endif
#if defined (USE_CUDA_GRAPH)
    checkCudaErrors(cudaFree(m_center_tmp_ref));
    checkCudaErrors(cudaFree(m_center_tmp_pos));
    checkCudaErrors(cudaFree(d_count_ref));
    checkCudaErrors(cudaFree(d_count_pos));
#else
    checkCudaErrors(cudaFree(m_center_tmp));
#endif
    checkCudaErrors(cudaFree(m_device_rmsd));
    checkCudaErrors(cudaFreeHost(m_host_rmsd));
    checkCudaErrors(cudaFree(d_count));
#if defined(USE_NR)
    checkCudaErrors(cudaFreeHost(max_reached));
#endif
#if defined (USE_CUDA_GRAPH)
    if (m_graph) checkCudaErrors(cudaGraphDestroy(m_graph));
    if (m_instance) checkCudaErrors(cudaGraphExecDestroy(m_instance));
#endif
}
