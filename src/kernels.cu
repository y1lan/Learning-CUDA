#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T *diagonal, T *trace, size_t steps) {
  __shared__ T smem[32];
  // thread id in block
  size_t tid = threadIdx.x;
  // global id in grid
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  T thread_sum{};
  for (size_t i = gid; i < steps; i += stride) {
    thread_sum += diagonal[i];
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
  if (h_input.size() != rows * cols) {
    return T{};
  }
  size_t steps = std::min(rows, cols);
  if (steps == 0) {
    return T{};
  }
  const T *data = h_input.data();
  const unsigned block_size = 256;
  const dim3 blockdim = block_size;
  const dim3 griddim = (steps + block_size - 1) / block_size;
  T *input;
  T *diagonal;
  T *output;
  cudaMalloc(&input, h_input.size() * sizeof(T));
  cudaMalloc(&diagonal, steps * sizeof(T));
  cudaMalloc(&output, griddim.x * sizeof(T));
  cudaMemcpy(input, h_input.data(), h_input.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  trace_load<T><<<griddim, blockdim>>>(input, diagonal, steps, cols);
  trace_kernel<T><<<griddim, blockdim>>>(diagonal, output, steps);
  T sum{};
  std::vector<T> host_output(griddim.x);
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

template <typename T> __device__ __forceinline__ T dev_max(T lhs, T rhs) {
  return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ half dev_max(half lhs, half rhs) {
  return __hgt(lhs, rhs) ? lhs : rhs;
}

/**
  threadIdx.x -> the id of q
  threadIdx.y -> the q_block_id of q
  blockIdx.x -> the id of q_head
  blockIdx.y -> the id of batch
  Q_BLOCK_SIZE == blockDim.x
**/
template <typename T, int KV_BLOCK_SIZE, int HEADDIM>
__global__ void q_flash(const T *q, const T *k, const T *v, T *o,
                        int batch_size, int target_seq_len, int src_seq_len,
                        int query_heads, int kv_heads, int head_dim,
                        bool is_causal) {
  int batch_id = blockIdx.y;
  int qhead_id = blockIdx.x;
  int kv_head_id = (qhead_id * kv_heads) / query_heads;
  int q_id = threadIdx.y * blockDim.x + threadIdx.x;
  if (q_id >= target_seq_len) {
    return;
  }
  // there are many warps so that the kv cache can be filled in.
  int warp = threadIdx.y;
  int warp_stride = blockDim.y;
  // blockDim.x == 32, so is warp size.
  int lane = threadIdx.x;
  // KV_BLOCK_SIZE should be the product of Q_BLOCK_SIZE;
  __shared__ T k_cache[KV_BLOCK_SIZE][HEADDIM];
  __shared__ T v_cache[KV_BLOCK_SIZE][HEADDIM];

  T s_ij[KV_BLOCK_SIZE] = {};
  T o_cache[HEADDIM] = {};
  T o_new[HEADDIM] = {};
  T m{}, l{};
  const T *q_offset = q + batch_id * (target_seq_len * query_heads * head_dim) +
                      q_id * (query_heads * head_dim) + qhead_id * head_dim;
  T *o_offset = o + batch_id * (target_seq_len * query_heads * head_dim) +
                q_id * (query_heads * head_dim) + qhead_id * head_dim;
  int kv_steps = 0;
  for (int kv_block = 0; kv_block < src_seq_len; kv_block += KV_BLOCK_SIZE) {
    // load just one block of kv data
    if (is_causal && kv_steps > q_id) {
      break;
    }
    for (int kv_cache_id = warp;
         kv_cache_id < KV_BLOCK_SIZE && kv_cache_id + kv_block < src_seq_len;
         kv_cache_id += warp_stride) {
      for (int head_dim_id = lane; head_dim_id < head_dim;
           head_dim_id += HEADDIM) {
        if (head_dim_id < head_dim) {
          k_cache[kv_cache_id][head_dim_id] =
              *(k + batch_id * (src_seq_len * kv_heads * head_dim) +
                (kv_block + kv_cache_id) * (kv_heads * head_dim) +
                kv_head_id * (head_dim) + head_dim_id);
          v_cache[kv_cache_id][head_dim_id] =
              *(v + batch_id * (src_seq_len * kv_heads * head_dim) +
                (kv_block + kv_cache_id) * (kv_heads * head_dim) +
                kv_head_id * (head_dim) + head_dim_id);
        }
      }
    }
    // load all kv data into kv cache
    __syncthreads();

    // S_ij = Q_i \times K_i^T \in R^{1 \times KV_BLOCK_SIZE}
    int block_steps = (kv_steps + KV_BLOCK_SIZE > src_seq_len)
                          ? src_seq_len - kv_steps
                          : KV_BLOCK_SIZE;
    for (int i = 0; i < block_steps; i++) {
      int global_kv_id = kv_steps + i;
      if (is_causal && global_kv_id > q_id) {
        s_ij[i] = -INFINITY;
      } else {
        s_ij[i] = T{};
#pragma unroll
        for (int j = 0; j < head_dim; j++) {
          s_ij[i] += (*(q_offset + j)) * (k_cache[i][j]);
        }
      }
    }
    // m_ij = rowmax(S_ij)
    T m_ij = s_ij[0];
#pragma unroll
    for (int i = 0; i < block_steps; i++) {
      m_ij = dev_max<T>(m_ij, s_ij[i]);
    }

    // P_ij = exp(S_ij - m_ij) \in R^{1 \times KV_BLOCK_SIZE}
    // l_ij = rowsum(P_ij) \in R;
    T l_ij = 0;
#pragma unroll
    for (int i = 0; i < block_steps; i++) {
      s_ij[i] = __expf((float)(s_ij[i] - m_ij));
      l_ij += s_ij[i];
    }

    T m_new = dev_max<T>(m, m_ij);
    T l_new = (T)__expf((float)(m - m_new)) * l +
              (T)__expf((float)(m_ij - m_new)) * l_ij;

#pragma unroll
    for (int i = 0; i < head_dim; i++) {
      o_cache[i] = o_cache[i] * (T)(__expf((float)(m - m_new))) * (l) / l_new;
    }

    // e^{m_ij - m_new} \dot P_ij \times V_j;
    for (int i = 0; i < head_dim; i++) {
      o_new[i] = T{};
      for (int j = 0; j < block_steps; j++) {
        o_new[i] += (double)s_ij[j] * (double)v_cache[j][i];
      }
      o_new[i] = o_new[i] * (T)(__expf((float)(m_ij - m_new))) / l_new;
    }

#pragma unroll
    for (int i = 0; i < head_dim; i++) {
      o_cache[i] = o_cache[i] + o_new[i];
    }
    l = l_new;
    m = m_new;
    kv_steps += block_steps;

    __syncthreads();
  }

  for (int i = 0; i < head_dim; i++) {
    *(o_offset + i) = o_cache[i];
  }
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

  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, h_q.size() * sizeof(T));
  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  cudaMalloc(&d_k, h_k.size() * sizeof(T));
  cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  cudaMalloc(&d_v, h_v.size() * sizeof(T));
  cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaMalloc(&d_o, h_o.size() * sizeof(T));
  cudaMemcpy(d_o, h_o.data(), h_o.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  // cudaStream_t stream[32];
  switch (head_dim) {
  case 1:
  case 2:
  case 4:
  case 8:
  case 16:
  case 32:
    q_flash<T, 64, 32><<<dim3{(unsigned)query_heads, (unsigned)batch_size},
                         dim3{32, ((unsigned)target_seq_len / 32) + 1}, 0, 0>>>(
        d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
    break;
  case 64:
    q_flash<T, 32, 64><<<dim3{(unsigned)query_heads, (unsigned)batch_size},
                         dim3{32, ((unsigned)target_seq_len / 32) + 1}, 0, 0>>>(
        d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
    break;
  }

  // for (int i = 0; i < 32; i++) {
  //   cudaStreamSynchronize(stream[i]);
  // }
  cudaDeviceSynchronize();
  cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
  return;
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
