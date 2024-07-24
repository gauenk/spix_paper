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


inline __host__ __device__
void check_valid(bool& valid, const int hi, const int wi, const int H, const int W){
  valid = (hi <= (H-1)) and (hi >= 0);
  valid = valid and (wi <= (W-1)) and (wi >= 0);
}


template <typename scalar_t>
__global__ void sna_gather_sims_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sims_out,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims_in,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds){

    // -- unpack --
    int nbatch = sims_in.size(0);
    int height = sims_in.size(1);
    int width = sims_in.size(2);
    int sH = sims_in.size(3);
    int sW = sims_in.size(4);
    int num_pix = height*width;

    // -- compute indices -- 
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int si = threadIdx.y;
    int ibatch = blockIdx.y;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster - hi * width;
    // int wi = hw_raster - hi * width;

    // -- read sims P(L_j = s) --
    int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    scalar_t sim_prob = valid ? sims_in[ibatch][hi][wi][s_hi][s_wi] : 0;

    // -- store along axis --
    sims_out[ibatch][hi][wi][si] = sim_prob;
}

void sna_gather_sims_forward_cuda(torch::Tensor sims_out,
                                   const torch::Tensor sims_in,
                                   const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(sims_out);
    CHECK_INPUT(sims_in);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = sims_in.size(0);
    int height = sims_in.size(1);
    int width = sims_in.size(2);
    int num_pix = height*width;
    int nsuperpixels = 9;

    // -- block --
    int nthreads_pix = 112;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    dim3 nthreads(nthreads_pix,nsuperpixels);
    dim3 nblock(nblocks_pix,nbatch);

    AT_DISPATCH_FLOATING_TYPES(sims_in.type(), "forward_kernel", ([&] {
        sna_gather_sims_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            sims_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            sims_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>());
        }));
}



/***********************************************


               Backward Kernel


 ***********************************************/

template <typename scalar_t>
__global__ void sna_gather_sims_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dsims_in,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dsims_out,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds){

    // -- unpack --
    int NSP_PTHRED = 2;
    int nbatch = dsims_in.size(0);
    int height = dsims_in.size(1);
    int width = dsims_in.size(2);
    int sH = dsims_in.size(3);
    int sW = dsims_in.size(4);
    int num_pix = height*width;

    // -- compute indices -- 
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int si_s = NSP_PTHRED*threadIdx.y;
    int ibatch = blockIdx.y;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster - hi * width;

    // -- read grad --
    for (int si_pt=0; si_pt< NSP_PTHRED; si_pt++){
      int si = si_s + si_pt;
      if (si >= 9){ return; }
      scalar_t dsim = dsims_out[ibatch][hi][wi][si];
  
      // -- read sims P(L_j = s) --
      int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
      int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
      bool valid = true;
      check_valid(valid,s_hi,s_wi,sH,sW);
      if (valid){
        dsims_in[ibatch][hi][wi][s_hi][s_wi] = dsim; 
      }
    }
}

void sna_gather_sims_backward_cuda(torch::Tensor dsims_in,
                                    const torch::Tensor dsims_out,
                                    const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(dsims_in);
    CHECK_INPUT(dsims_out);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = dsims_in.size(0);
    int height = dsims_in.size(1);
    int width = dsims_in.size(2);
    int num_pix = height*width;
    int nsuperpixels = 9;

    // -- block --
    // int nthreads_pix = 112;
    int nthreads_pix = 192;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    dim3 nthreads(nthreads_pix,(nsuperpixels-1)/2+1);
    dim3 nblock(nblocks_pix,nbatch);
    AT_DISPATCH_FLOATING_TYPES(dsims_in.type(), "backward_kernel", ([&] {
        sna_gather_sims_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            dsims_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dsims_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>());
        }));
}


void init_gather_sims(py::module &m){
  m.def("gather_sims_forward",&sna_gather_sims_forward_cuda,
        "gather P(L_i=s) forward");
  m.def("gather_sims_backward",&sna_gather_sims_backward_cuda,
        "gather P(L_i=s) backward");
}
