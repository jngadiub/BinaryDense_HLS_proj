//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"


int main(int argc, char **argv)
{

  //hls-fpga-machine-learning insert data
  input_t  data_str[N_INPUTS] = {0.375406, 0.904037, 0.0774234, 0.00648763, 0.0749703, 0.00445173, 0.346424, 0.15876, 0.346424, 0.755111, 0.7699, 0.600613, 0.824538, 0.601524, 0.0363961, 0.219895};


  result_t res_str[N_OUTPUTS] = {0};
  unsigned short size_in, size_out;
  myproject(data_str, res_str, size_in, size_out);
    
  for(int i=0; i<N_OUTPUTS; i++){
    std::cout << res_str[i] << " ";
  }
  std::cout << std::endl;
  
  return 0;
}
