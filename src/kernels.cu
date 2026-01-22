#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T *diagonol, T *trace, size_t steps) {
  __shared__ T smem[32];
  // thread id in block
  size_t tid = threadIdx.x;
  // global id in grid
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  T thread_sum{};
  for (size_t i = gid; i < steps; i += stride) {
    thread_sum += diagonol[i];
  }
  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(mask, thread_sum, offset);
  }
  // offset in wrap
  size_t lane_id = tid & 31;
  // wrap id in block
  size_t wrap_id = tid >> 5;
  if (lane_id == 0) {
    smem[wrap_id] = thread_sum;
  }
  __syncthreads();

  // only wrap_id == 0 perform the following instructions;
  T val = (tid < blockDim.x / 32) ? smem[lane_id] : T{};

  if (wrap_id == 0) {
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(mask, val, offset);
    }
    if (lane_id == 0) {
      trace[blockIdx.x] = val;
    }
  }
}

template <typename T>
__global__ void trace_load(const T *data, T *diagnol, size_t steps,
                           size_t cols) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < steps; i += stride) {
    diagnol[i] = data[i * cols + i];
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols) {
  if(h_input.size() != rows * cols){
    return T{};
  }
  size_t steps = std::min(rows, cols);
  if (steps == 0){
    return T{};
  }
  const T *data = h_input.data();
  const unsigned block_size = 256;
  const dim3 blockdim = block_size;
  const dim3 griddim = (steps + block_size - 1)/ block_size;
  T *input;
  T* diagonal;
  T *output;
  cudaMalloc(&input, h_input.size() * sizeof(T));
  cudaMalloc(&diagonal, steps * sizeof(T));
  cudaMalloc(&output, griddim.x * sizeof(T));
  cudaMemcpy(input, h_input.data(), h_input.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  
  trace_load<T><<<griddim, blockdim>>>(input, diagonal, steps, cols);
  trace_kernel<T><<<griddim, blockdim>>>(diagonal, output, steps);
  T sum{};
  std::vector<T>host_output(griddim.x);
  cudaMemcpy(host_output.data(), output, griddim.x * sizeof(T),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("trace_kernel error: %s\n", cudaGetErrorString(err));
    cudaFree(input);
    cudaFree(diagonal);
    cudaFree(output);
    return T{};
  }
  for (size_t i = 0; i < griddim.x; i++) {
    sum += host_output[i];
  }
  cudaFree(input);
  cudaFree(output);
  cudaFree(diagonal); 
  return sum;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
                                   const std::vector<half> &,
                                   const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int,
                                   int, bool);
