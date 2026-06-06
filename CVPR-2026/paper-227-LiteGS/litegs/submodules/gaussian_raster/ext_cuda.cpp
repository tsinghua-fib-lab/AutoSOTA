#include <torch/extension.h>
#include "binning.h"
#include "compact.h"
#include "raster.h"
#include "transform.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("duplicateWithKeys", &duplicateWithKeys);
  m.def("tileRange", &tileRange);
  m.def("create_ROI_AABB", &create_ROI_AABB);
  m.def("rasterize_forward", &rasterize_forward);
  m.def("rasterize_backward", &rasterize_backward);
  m.def("jacobianRayspace", &jacobianRayspace);
  m.def("createTransformMatrix_forward", &createTransformMatrix_forward);
  m.def("createTransformMatrix_backward", &createTransformMatrix_backward);
  m.def("world2ndc_backword", &world2ndc_backword);
  m.def("createCov2dDirectly_forward", &createCov2dDirectly_forward);
  m.def("createCov2dDirectly_backward", &createCov2dDirectly_backward);
  m.def("sh2rgb_forward", &sh2rgb_forward);
  m.def("sh2rgb_backward", &sh2rgb_backward);
  m.def("eigh_and_inv_2x2matrix_forward", &eigh_and_inv_2x2matrix_forward);
  m.def("inv_2x2matrix_backward", &inv_2x2matrix_backward);
  m.def("compact_visible_params_forward", &compact_visible_params_forward);
  m.def("compact_visible_params_backward", &compact_visible_params_backward);
  m.def("adamUpdate", &adamUpdate);
}