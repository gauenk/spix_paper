#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>


#define CUDA_NUM_THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// inline __host__ __device__ int get_window_start(
//     const int index,
//     const int length,
//     const int KERNEL_SIZE,
//     const int NEIGHBORHOOD_SIZE) {
//   int dilation = 1;
//   if (dilation <= 1)
//     return max(index - NEIGHBORHOOD_SIZE, 0) +
//         (index + NEIGHBORHOOD_SIZE >= length) *
//         (length - index - NEIGHBORHOOD_SIZE - 1);
//   int ni = index - NEIGHBORHOOD_SIZE * dilation;
//   if (ni < 0)
//     return index % dilation;
//   if (index + NEIGHBORHOOD_SIZE * dilation >= length) {
//     const int imodd = index % dilation;
//     const int a = int(length / dilation) * dilation;
//     const int b = length - a;
//     if (imodd < b)
//       return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
//     return a + imodd - KERNEL_SIZE * dilation;
//   }
//   return ni;
// }

inline __host__ __device__
int get_window_start(const int index, const int length, const int neigh_size){
  int new_index = max(index - neigh_size,0);
  new_index += ((index+neigh_size)>=length) * (length - index - neigh_size - 1);
  return new_index;
}

inline __host__ __device__
void check_valid(bool& valid, const int hi, const int wi, const int H, const int W){
  valid = (hi <= (H-1)) and (hi >= 0);
  valid = valid and (wi <= (W-1)) and (wi >= 0);
}


template <typename scalar_t>
__global__ void sna_reweight_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn_out,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn_in,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size, int NUM_PER_THREAD){

    // -- unpack --
    int nbatch = attn_in.size(0);
    int nheads = attn_in.size(1);
    int height = attn_in.size(2);
    int width = attn_in.size(3);
    int ksize_sq = attn_in.size(4);
    int num_pix = height*width;
    int sH = sims.size(3);
    int sW = sims.size(4);

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    // int ihead = blockIdx.z - ibatch * nheads;
    int ihead = blockIdx.z % nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset_s = NUM_PER_THREAD*(blockIdx.y * blockDim.y + threadIdx.y);
    int si = threadIdx.z;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }
    // if (attn_offset_s >= ksize_sq){ return; }

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster % width;
    int neigh_size = (kernel_size-1)/2;

    // -- read sims P(L_j = s) --
    int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    int h_start = get_window_start(hi, height, neigh_size);
    int w_start = get_window_start(wi, width, neigh_size);
    // scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : 0;

    // -- accumulate --
    for (int _idx=0; _idx < NUM_PER_THREAD; _idx++){

      // -- derivative indices --
      int attn_offset = attn_offset_s + _idx;
      if (attn_offset >= ksize_sq){ return; }
      int h_offset = attn_offset / kernel_size;
      int w_offset = attn_offset % kernel_size;
      int v_hi = h_start+h_offset;
      int v_wi = w_start+w_offset;
      scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : 0;
      // sim_prob = v_wi;
      // sim_prob = 1;

      scalar_t attn_val = attn_in[ibatch][ihead][hi][wi][attn_offset];
      atomicAdd(&(attn_out[ibatch][ihead][si][hi][wi][attn_offset]),sim_prob*attn_val);
    }
}

void sna_reweight_forward_cuda(torch::Tensor attn_out,
                                const torch::Tensor attn_in,
                                const torch::Tensor sims,
                                const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(attn_in);
    CHECK_INPUT(attn_out);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = attn_in.size(0);
    int nheads = attn_in.size(1);
    int height = attn_in.size(2);
    int width = attn_in.size(3);
    int ksize_sq = attn_in.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;
    int nsuperpixels = 9;
    int NUM_PER_THREAD = 4;

    // -- block --
    int nthreads_pix = 14;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/(NUM_PER_THREAD*nthreads_ksize)+1;
    // int nblocks_ksize = (ksize_sq-1)/(NUM_PER_THREAD*nthreads_ksize)+10;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nsuperpixels);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn_in.type(), "forward_kernel", ([&] {
        sna_reweight_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            attn_out.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            attn_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size,NUM_PER_THREAD);
        }));
}



/***********************************************


               Backward Kernel


 ***********************************************/

template <typename scalar_t>
__global__ void sna_reweight_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_attn_in,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_sims,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> d_attn_out,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn_out,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn_in,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size, int NUM_PER_THREAD){

    // -- unpack --
    int nbatch = d_attn_in.size(0);
    int nheads = d_attn_in.size(1);
    int height = d_attn_in.size(2);
    int width = d_attn_in.size(3);
    int ksize_sq = d_attn_in.size(4);
    int num_pix = height*width;
    int sH = sims.size(3);
    int sW = sims.size(4);

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    // int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int attn_offset_s = NUM_PER_THREAD*(blockIdx.y * blockDim.y + threadIdx.y);
    int si = threadIdx.z;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }

    // -- derivative indices --
    int neigh_size = (kernel_size-1)/2;

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster - hi * width;
    int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    int h_start = get_window_start(hi, height, neigh_size);
    int w_start = get_window_start(wi, width, neigh_size);

    // -- accumulate --
    for (int _idx=0; _idx < NUM_PER_THREAD; _idx++){

      // -- indices --
      int attn_offset = attn_offset_s + _idx;
      if (attn_offset >= ksize_sq){ return; }
      int h_offset = attn_offset / kernel_size;
      int w_offset = attn_offset % kernel_size;
      int v_hi = h_start+h_offset;
      int v_wi = w_start+w_offset;

      // -- read sims --
      scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : 0;

      // -- derivatives ("l,i" = pixel idx ,"f" = feature idx, "j" = attn map idx) --
      scalar_t attn_val = attn_in[ibatch][ihead][hi][wi][attn_offset];
      scalar_t d_attn_val = d_attn_out[ibatch][ihead][si][hi][wi][attn_offset];

      // -- derivatives --
      if (valid){
        atomicAdd(&(d_sims[ibatch][v_hi][v_wi][s_hi][s_wi]),d_attn_val*attn_val);
        atomicAdd(&(d_attn_in[ibatch][ihead][hi][wi][attn_offset]),d_attn_val*sim_prob);
      }

    }
}

void sna_reweight_backward_cuda(torch::Tensor d_attn_in,
                                 torch::Tensor d_sims,
                                 const torch::Tensor d_attn_out,
                                 const torch::Tensor attn_out,
                                 const torch::Tensor attn_in,
                                 const torch::Tensor sims,
                                 const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(d_attn_in);
    CHECK_INPUT(d_sims);
    CHECK_INPUT(d_attn_out);
    CHECK_INPUT(attn_out);
    CHECK_INPUT(attn_in);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = d_attn_in.size(0);
    int nheads = d_attn_in.size(1);
    int height = d_attn_in.size(2);
    int width = d_attn_in.size(3);
    int ksize_sq = d_attn_in.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;
    int nsuperpixels = 9;
    int NUM_PER_THREAD = 4;

    // -- block --
    int nthreads_pix = 14;
    int nthreads_ksize = 8;
    // int nthreads_pix = 12;
    // int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/(NUM_PER_THREAD*nthreads_ksize)+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nsuperpixels);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(d_attn_in.type(), "backward_kernel", ([&] {
        sna_reweight_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            d_attn_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_attn_out.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            attn_out.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            attn_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size,NUM_PER_THREAD);
        }));

}


void init_sna_reweight(py::module &m){
  m.def("reweight_forward", &sna_reweight_forward_cuda,
        "neighborhood superpixel atten forward");
  m.def("reweight_backward", &sna_reweight_backward_cuda,
        "neighborhood superpixel atten backward");
}
