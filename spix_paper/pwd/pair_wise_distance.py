
import torch
import st_spix_cuda

class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, pixel_ftrs, spixel_ftrs, init_spixel_indices,
                num_spixels_width, num_spixels_height):
        self.num_spixels_width = num_spixels_width
        self.num_spixels_height = num_spixels_height
        output = pixel_ftrs.new(pixel_ftrs.shape[0],
                                9, pixel_ftrs.shape[-1]).zero_()
        self.save_for_backward(pixel_ftrs, spixel_ftrs, init_spixel_indices)
        return st_spix_cuda.pwd_forward(
            pixel_ftrs.contiguous(), spixel_ftrs.contiguous(),
            init_spixel_indices.contiguous(), output,
            self.num_spixels_width, self.num_spixels_height)

    @staticmethod
    def backward(self, dist_matrix_grad):
        pixel_ftrs, spixel_ftrs, init_spixel_indices = self.saved_tensors

        pixel_ftrs_grad = torch.zeros_like(pixel_ftrs)
        spixel_ftrs_grad = torch.zeros_like(spixel_ftrs)

        # pair_wise_distance_cuda
        pixel_ftrs_grad, spixel_ftrs_grad = st_spix_cuda.pwd_backward(
            dist_matrix_grad.contiguous(), pixel_ftrs.contiguous(),
            spixel_ftrs.contiguous(), init_spixel_indices.contiguous(),
            pixel_ftrs_grad, spixel_ftrs_grad,
            self.num_spixels_width, self.num_spixels_height
        )
        return pixel_ftrs_grad, spixel_ftrs_grad, None, None, None

