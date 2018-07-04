#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_batchnorm.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
typedef ap_fixed<18,8> accum_inputs_default_t;
typedef ap_int<8> accum_default_t;
typedef ap_int<2> mult_default_t;
typedef ap_fixed<18,8> mult_inputs_default_t;
typedef ap_int<2> weight_default_t;
typedef ap_fixed<18,8> bias_default_t;
typedef ap_fixed<18,8> beta_default_t;
typedef ap_fixed<18,8> mean_default_t;
typedef ap_fixed<18,8> scale_default_t;
#define N_INPUTS 16
#define N_LAYER_1 64
#define N_LAYER_2 64
#define N_LAYER_3 64
#define N_LAYER_4 32
#define N_LAYER_5 32
#define N_LAYER_6 32
#define N_LAYER_7 32
#define N_LAYER_8 32
#define N_LAYER_9 32
#define N_LAYER_10 5
#define N_OUTPUTS 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,8> layer1_t;
typedef ap_fixed<18,8> layer2_t;
typedef ap_int<2> layer3_t;
typedef ap_int<8> layer4_t;
typedef ap_fixed<18,8> layer5_t;
typedef ap_int<2> layer6_t;
typedef ap_int<8> layer7_t;
typedef ap_fixed<18,8> layer8_t;
typedef ap_int<2> layer9_t;
typedef ap_int<8> layer10_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::layer_config {
        static const unsigned n_in = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_inputs_default_t accum_t;
        typedef mult_inputs_default_t mult_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config1 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config2 : nnet::batchnorm_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_LAYER_2;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool store_weights_in_bram = false;
        typedef accum_inputs_default_t accum_t;
        typedef beta_default_t beta_t;
        typedef scale_default_t scale_t;
        typedef mean_default_t mean_t;
        };
struct binary_tanh_config3 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_2;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config4 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_3;
        static const unsigned n_out = N_LAYER_4;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef mult_default_t mult_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config4 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config5 : nnet::batchnorm_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned n_out = N_LAYER_5;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool store_weights_in_bram = false;
        typedef accum_inputs_default_t accum_t;
        typedef beta_default_t beta_t;
        typedef scale_default_t scale_t;
        typedef mean_default_t mean_t;
        };
struct binary_tanh_config6 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_5;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config7 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_6;
        static const unsigned n_out = N_LAYER_7;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef mult_default_t mult_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config7 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_7;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config8 : nnet::batchnorm_config {
        static const unsigned n_in = N_LAYER_7;
        static const unsigned n_out = N_LAYER_8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool store_weights_in_bram = false;
        typedef accum_inputs_default_t accum_t;
        typedef beta_default_t beta_t;
        typedef scale_default_t scale_t;
        typedef mean_default_t mean_t;
        };
struct binary_tanh_config9 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config10 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_9;
        static const unsigned n_out = N_LAYER_10;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef mult_default_t mult_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config10 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_10;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config11 : nnet::batchnorm_config {
        static const unsigned n_in = N_LAYER_10;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool store_weights_in_bram = false;
        typedef accum_inputs_default_t accum_t;
        typedef beta_default_t beta_t;
        typedef scale_default_t scale_t;
        typedef mean_default_t mean_t;
        };

#endif 
