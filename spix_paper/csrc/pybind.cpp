/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>

// -- fxns --
// void init_bass(py::module &);
void init_sna_reweight(py::module &);
void init_pwd(py::module &);
void init_gather_sims(py::module &m);
// void init_scatter_img(py::module &);
// void init_scatter_spix(py::module &);
// void init_spix_prop_dev(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // init_bass(m);
  // init_scatter_img(m);
  // init_scatter_spix(m);
  // init_spix_prop_dev(m);
  init_sna_reweight(m);
  init_pwd(m);
  init_gather_sims(m);
}
