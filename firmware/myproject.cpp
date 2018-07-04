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
#include <iostream>

#include "parameters.h"
#include "myproject.h"

#include "nnet_layer.h"
#include "nnet_batchnorm.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/beta2.h"
#include "weights/scale2.h"
#include "weights/mean2.h"
#include "weights/w4.h"
#include "weights/beta5.h"
#include "weights/scale5.h"
#include "weights/mean5.h"
#include "weights/w7.h"
#include "weights/beta8.h"
#include "weights/scale8.h"
#include "weights/mean8.h"
#include "weights/w10.h"
#include "weights/beta11.h"
#include "weights/scale11.h"
#include "weights/mean11.h"

void myproject(
		  input_t data[N_INPUTS],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = N_INPUTS;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[N_LAYER_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    layer1_t logits1[N_LAYER_1];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::compute_layer_nobias<input_t, layer1_t, config1>(data, logits1, w1);
    nnet::linear<layer1_t, layer1_t, linear_config1>(logits1, layer1_out);

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    layer2_t logits2[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::normalize<layer1_t, layer2_t, config2>(layer1_out, logits2, scale2, beta2, mean2);

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::binary_tanh<layer2_t, layer3_t, binary_tanh_config3>(logits2, layer3_out);

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    layer4_t logits4[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=logits4 complete dim=0
    nnet::compute_layer_nobias<layer3_t, layer4_t, config4>(layer3_out, logits4, w4);
    nnet::linear<layer4_t, layer4_t, linear_config4>(logits4, layer4_out);

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    layer5_t logits5[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=logits5 complete dim=0
    nnet::normalize<layer4_t, layer5_t, config5>(layer4_out, logits5, scale5, beta5, mean5);

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::binary_tanh<layer5_t, layer6_t, binary_tanh_config6>(logits5, layer6_out);

    layer7_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    layer7_t logits7[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=logits7 complete dim=0
    nnet::compute_layer_nobias<layer6_t, layer7_t, config7>(layer6_out, logits7, w7);
    nnet::linear<layer7_t, layer7_t, linear_config7>(logits7, layer7_out);

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    layer8_t logits8[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=logits8 complete dim=0
    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, logits8, scale8, beta8, mean8);

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::binary_tanh<layer8_t, layer9_t, binary_tanh_config9>(logits8, layer9_out);

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    layer10_t logits10[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=logits10 complete dim=0
    nnet::compute_layer_nobias<layer9_t, layer10_t, config10>(layer9_out, logits10, w10);
    nnet::linear<layer10_t, layer10_t, linear_config10>(logits10, layer10_out);

    nnet::normalize<layer10_t, result_t, config11>(layer10_out, res, scale11, beta11, mean11);


}
