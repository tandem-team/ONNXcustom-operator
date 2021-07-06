#include <fstream>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>

torch::Tensor repeatInterleave(torch::Tensor input) 
{
  
  int size_desired = 512;
  int size_present = input.sizes()[1];
  //float data_array[512] = {0};
  
  torch::Tensor output_array = torch::zeros({1, 512});
  
  int interleave_factor = size_desired/size_present;
  int iterator=512;
  
  
  
  int index=0;
  int size_present_iter=0;
  int temp=0;
  while(iterator>0)
  {
  
    for(index=0;index<interleave_factor;index++){
      temp=512-iterator;
      output_array[0][temp+index]=input[0][size_present_iter];
    }
  
    size_present_iter++;
    iterator = iterator-interleave_factor;
  }
  
  return output_array;
}



torch::Tensor reduction (
 torch::Tensor layerOne,
 torch::Tensor layerTwo,
 torch::Tensor layerThree,
 torch::Tensor layerFour) {

torch::Tensor layerOne_n, layerTwo_n, layerThree_n, layerFour_n;

layerOne_n = repeatInterleave(layerOne);
layerTwo_n = repeatInterleave(layerTwo);
layerThree_n = repeatInterleave(layerThree);
layerFour_n = repeatInterleave(layerFour);


return layerOne_n;
}


int main() {
//torch::Tensor layerTwo_n, layerThree_n, layerFour_n;

int initializer[64];


int i=0;

for(i=0;i<64;i++)
initializer[i] = i;

std::vector<int> data_array(initializer, initializer+sizeof(initializer)/sizeof(initializer[0]));

torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

torch::Tensor layerOne_n = torch::from_blob(initializer, {1, 64}, torch::kInt);
std::cout << "The initial input is :" << std::endl;
std::cout << layerOne_n[0] << std::endl;


torch::Tensor output =  repeatInterleave(layerOne_n);
std::cout << "The final output is :" << std::endl;
std::cout << output[0] << std::endl;


std::cout << "CUDA Value is : " << torch::cuda::is_available() << device << std::endl;

return 0;
}

