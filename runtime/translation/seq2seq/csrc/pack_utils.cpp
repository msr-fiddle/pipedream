#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <torch/torch.h>

namespace at { namespace native {

at::Tensor revert_varlen_tensor(const Tensor& input, const Tensor& lengths);

void checkLongTensor(const Tensor& tensor);


at::Tensor set_mask_cpp(const Tensor& _lengths){
   at::native::checkLongTensor(_lengths);
   int64_t batch_size = _lengths.size(0);
   int64_t * lengths = _lengths.data<int64_t>();
   int64_t seq_length = (lengths == NULL) ? 0 : lengths[0];
   auto output = torch::empty({seq_length, batch_size}, torch::CPU(at::kByte));
   auto output_data = output.data<uint8_t>();
   for (int64_t t = 0; t< seq_length; t++){
       for (int64_t i = 0; i < batch_size; i++){
           if (lengths[i] > t) {
               output_data[t*batch_size + i] = 1;
           } else {
               output_data[t*batch_size + i] = 0;
           }
    }
   } 
   return output;

}   

}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("revert_varlen_tensor", &at::native::revert_varlen_tensor);
  m.def("set_mask_cpp", &at::native::set_mask_cpp);
}
