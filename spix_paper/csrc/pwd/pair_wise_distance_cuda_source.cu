#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 256

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ dist_matrix,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;

    int init_spix_index = spixel_indices[b * num_pixels + p];

    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);

    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = 1e16;
    }
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h){
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = 1e16;
    }
    else {
        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;

        scalar_t sum_squared_diff = 0;
        for (int c=0; c<channels; c++)
        {
            sum_squared_diff += pow(pixel_features[b * cp + c * num_pixels + p] - 
                spixel_features[b * cs + c * num_spixels + query_spixel_index], 2);
        }
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = sum_squared_diff;
    }
}

torch::Tensor pwd_forward_cuda(
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor spixel_indices,
    torch::Tensor dist_matrix,
    int num_spixels_w, int num_spixels_h){

    // -- check --
    CHECK_CUDA(pixel_features);
    CHECK_CUDA(spixel_features);
    CHECK_CUDA(spixel_indices);
    CHECK_CUDA(dist_matrix);
  
    // -- unpack --
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);

    // -- allocate threads --
    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    // -- info --
    // fprintf(stdout,"batchsize,channels: %d,%d\n",batchsize,channels);
    // fprintf(stdout,"num_pixels,num_spixels,num_pixels_w,num_pixels_h: %d,%d,%d,%d\n",
    //         num_pixels,num_spixels,num_spixels_w,num_spixels_h);
    // fprintf(stdout,"ntotal: %d\n",
    //         batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1);
    // fprintf(stdout,"nblocks: %d\n",
    //         (batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    // -- launch cuda --
    AT_DISPATCH_FLOATING_TYPES(dist_matrix.type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            pixel_features.data<scalar_t>(),
            spixel_features.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            dist_matrix.data<scalar_t>(),
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h);
    }));
    return dist_matrix;
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ dist_matrix_grad,
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ pixel_feature_grad,
    scalar_t* __restrict__ spixel_feature_grad,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;

    int init_spix_index = spixel_indices[b * num_pixels + p];

    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);

    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) return;
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) return;
    else {
        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;

        scalar_t dist_matrix_grad_val = dist_matrix_grad[b * (9 * num_pixels) + spixel_offset * num_pixels + p];

        for (int c=0; c<channels; c++)
        {
            scalar_t pix_value = pixel_features[b * cp + c * num_pixels + p];
            scalar_t spix_value = spixel_features[b * cs + c * num_spixels + query_spixel_index];
            scalar_t diff = (pix_value - spix_value) * dist_matrix_grad_val;
            atomicAdd(&pixel_feature_grad[b * cp + c * num_pixels + p], 2 * diff);
            atomicAdd(&spixel_feature_grad[b * cs + c * num_spixels + query_spixel_index], -2 * diff);
        }
    }
}


std::vector<torch::Tensor> pwd_backward_cuda(
    const torch::Tensor dist_matrix_grad,
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor spixel_indices,
    torch::Tensor pixel_features_grad,
    torch::Tensor spixel_features_grad,
    int num_spixels_w, int num_spixels_h){

    // -- check --
    CHECK_CUDA(dist_matrix_grad);
    CHECK_CUDA(pixel_features);
    CHECK_CUDA(spixel_features);
    CHECK_CUDA(spixel_indices);
    CHECK_CUDA(pixel_features_grad);
    CHECK_CUDA(spixel_features_grad);

    // -- unpack --
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);

    // -- launch cuda --
    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    AT_DISPATCH_FLOATING_TYPES(pixel_features_grad.type(), "backward_kernel", ([&] {
        backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            dist_matrix_grad.data<scalar_t>(),
            pixel_features.data<scalar_t>(),
            spixel_features.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            pixel_features_grad.data<scalar_t>(),
            spixel_features_grad.data<scalar_t>(),
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h
        );
    }));

    return {pixel_features_grad, spixel_features_grad};
}

void init_pwd(py::module &m){
  m.def("pwd_forward", &pwd_forward_cuda, "pair_wise_distance forward");
  m.def("pwd_backward", &pwd_backward_cuda, "pair_wise_distance backward");
}


