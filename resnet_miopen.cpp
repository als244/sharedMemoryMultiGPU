#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <pthread.h>

#include "resnet_miopen.h"

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024
#define TILE_WIDTH 32
#define BLOCK_ROWS 8
#define CUDA_BATCH_SIZE 32
#define MAX_SHARED_MEMORY 48000
#define MAX_SHARED_MEM_FLOATS 12000
#define MAX_THREAD_PER_BLOCK_INCL_REG 512

#define MAX_REQ_CONV_FIND_ALGO 5

#define TO_PRINT false

/* DECLARING FUNCTIONS HERE */
void testConvolution(int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output);


/* START OF KERNELS/FUNCTIONS */

__global__ void setVal(int size, float val, float *out){
 	int ind = blockDim.x * blockIdx.x + threadIdx.x;
 	if (ind >= size){
 		return;
 	}
 	out[ind] = val;
}

void init_weights_gaussian_device(hiprandGenerator_t * gen, int size, float *X, float mean, float var){
 	float stddev = sqrtf(var);
 	hiprandStatus_t status = hiprandGenerateNormal(*gen, X, (size_t) size, mean, stddev);
 }


/* NON-OPTIMIZED CUSTOM KERNELS (non-bottleneck) */

// ASSUME 1-D launch
__global__ void addVec(int size, float * A, float * B, float * out){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size){
		return;
	}
	out[i] = A[i] + B[i];
}

__global__ void subVec(int size, float * A, float * B, float * out){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size){
		return;
	}
	out[i] = A[i] - B[i];
}

// GRID has dim (ROWS / TILE_WIDTH, COLS/TILE_WIDTH)
// each BLOCK has dim (TILE_WIDTH, TILE_WIDTH)
__global__ void matMul(const float *M, const float *N, int m, int k, int n, float *out){

	
	int row_ind = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int col_ind = blockIdx.y * TILE_WIDTH + threadIdx.y;

	if (row_ind >= m || col_ind >= n){
		return;
	}

	float val = 0;
	for (int z = 0; z < k; z++){
		val += M[row_ind * k + z] * N[z * n + col_ind];
	}
	out[row_ind * n + col_ind] = val;
}

// grid has dim (ROWS / TILE_WIDTH, COLS/TILE_WIDTH)
// each BLOCK has dim (TILE_WIDTH , TILE_WIDTH) = # of threads
__global__ void transpose(const float *in, int rows, int cols, float * out) {

  int row_ind = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int col_ind = blockIdx.y * TILE_WIDTH + threadIdx.y;
  
  
  if (col_ind >= cols || row_ind >= rows){
  	return;
  }

  out[col_ind * rows + row_ind] = in[row_ind * cols + col_ind];
}

// assume grid launch of (SPATIAL_OUT_DIM, SPATIAL_OUT_DIM) and block dim of (FILTERS)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void doMaxPool(const float * input, int kern_dim, int stride, int batch_size, int * max_inds, float * out){

	int filter_id = threadIdx.x;

	// know this because of launch specification
	int filters = blockDim.x;
	int in_spatial_dim = stride * gridDim.x;
	int out_spatial_dim = gridDim.x;

	int spatial_row_start = stride * blockIdx.x;
	int spatial_col_start = stride * blockIdx.y;

	int half_kernel_dim = kern_dim / 2;

	float max_val, inp_val;
	int spatial_row, spatial_col, max_ind, inp_ind, out_ind;
	for (int s = 0; s < batch_size; s++){
		max_val = -1024;
		max_ind = -1024;
		for (int row_off = -half_kernel_dim; row_off <= half_kernel_dim; row_off++){
			for (int col_off = -half_kernel_dim; col_off <= half_kernel_dim; col_off++){
				spatial_row = spatial_row_start + row_off;
				spatial_col = spatial_col_start + col_off;
				if ((spatial_row < 0) || (spatial_row >= in_spatial_dim) || (spatial_col < 0) || (spatial_col >= in_spatial_dim)){
					continue;
				}
				inp_ind = s * in_spatial_dim * in_spatial_dim * filters + filter_id * in_spatial_dim * in_spatial_dim + spatial_row * in_spatial_dim + spatial_col;
				inp_val = input[inp_ind];
				if (inp_val > max_val){
					max_val = inp_val;
					max_ind = inp_ind;
				}
			}
		}
		out_ind = s * filters * out_spatial_dim * out_spatial_dim + filter_id * out_spatial_dim * out_spatial_dim + blockIdx.x * out_spatial_dim + blockIdx.y;
		max_inds[out_ind] = max_ind;
		out[out_ind] = max_val;
	}
}

// assume grid launch of (SPATIAL_OUT_DIM, SPATIAL_OUT_DIM, OUT_FILTERS) and block dim of (BATCH_SIZE)
// max_inds_populated is mapping from max_pool_out_index -> associated max_index of input (populated from forward pass)
// also assume max_pool_inp_deriv is populated with all 0's to begin with and we overwrite non-zero values
__global__ void maxPoolDeriv(const int *max_inds_populated, const float *out_deriv, int kern_dim, int in_spatial_dim, int stride, int filters, int batch_size, float * max_pool_inp_deriv){

	int out_spatial_dim = in_spatial_dim / stride;

	int out_spatial_row = blockIdx.x;
	int out_spatial_col = blockIdx.y;
	int out_filter_id = blockIdx.z;
	int sample_ind = threadIdx.x;

	// based on launch spec should be ok, but check anyways
	if ((out_spatial_row >= out_spatial_dim) || (out_spatial_col >= out_spatial_dim) || (out_filter_id >= filters) || (sample_ind >= batch_size)){
		return;
	}

	int out_ind = sample_ind * filters * out_spatial_dim * out_spatial_dim + out_filter_id * out_spatial_dim * out_spatial_dim + out_spatial_row * out_spatial_dim + out_spatial_col;
	int max_ind_for_out = max_inds_populated[out_ind];

	max_pool_inp_deriv[max_ind_for_out] = out_deriv[out_ind];
}


// assume grid launch of (# Filters) and block dim of (batch size)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void doFilterAvgPool(const float * input, int spatial_dim, float * out){

	int filter_id = blockIdx.x;
	int sample_ind = threadIdx.x;

	// know this because of launch specification
	int filters = gridDim.x;

	float sum = 0;
	for (int row = 0; row < spatial_dim; row++){
		for (int col = 0; col < spatial_dim; col++){
			sum += input[sample_ind * filters * spatial_dim * spatial_dim + filter_id * spatial_dim * spatial_dim + row * spatial_dim + col];
		}
	}

	float avg_val = sum / (spatial_dim * spatial_dim);
	out[filters * sample_ind + filter_id] = avg_val;
}

// assume grid launch of (# Filters) and block dim of (batch size)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void filterAvgPoolDeriv(const float * pooled_deriv, int filters, int batch_size, int spatial_dim, float * out){

	int filter_id = blockIdx.x;
	int sample_ind = threadIdx.x;

	// unnecessary because of launch conditions, but putting anyways...
	if ((filter_id >= filters) || (sample_ind >= batch_size)){
		return;
	}

	// indexing into (N, 2048) = (batch_size, filters) matrix 
	float pooled_filt_deriv = pooled_deriv[sample_ind * filters + filter_id];
	float avg_pooled_filt_deriv = pooled_filt_deriv / (spatial_dim * spatial_dim);

	// populating the pre-pooled conv block output
	for (int row = 0; row < spatial_dim; row++){
		for (int col = 0; col < spatial_dim; col++){
			out[sample_ind * filters * spatial_dim * spatial_dim + filter_id * spatial_dim * spatial_dim + row * spatial_dim + col] = avg_pooled_filt_deriv;
		}
	}
}

__global__ void doActivation(int size, float * input, float * output){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	output[i] = fmaxf(0, input[i]);
}


__global__ void doActivationDeriv(int size, float *input, float * upstream_deriv, float * output){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	if (input[i] > 0){
		output[i] = upstream_deriv[i];
	}
	else{
		output[i] = 0;
	}
}

// assume pass in 1-D block with batch size blocks and 1 thread per block
// could exploit more parallelism here but shouldnt be bottleneck for now...
// assume X is a matrix where # rows = batch size and # columns = output dim
__global__ void softMax(const float * X, int batch_size, int output_len, float * out){
  int i = threadIdx.x;
  if (i < batch_size){
  	float max = X[i * output_len];
  	for (int j = 0; j < output_len; j++){
  		if (X[i * output_len + j] > max){
  			max = X[i * output_len + j];
  		}
  	}
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += expf(X[i * output_len + j] - max);
    }
    for (int j = 0; j < output_len; j++){
      out[i * output_len + j] = expf(X[i * output_len + j] - max) / sum;
    }
  }
}

// launch with gridDim (output_dim) and threadDim (batch_size)
__global__ void averageDerivOverBatchSize(float * output_deriv, int output_dim, int batch_size){

	int output_class = blockIdx.x;
	int sample_ind = threadIdx.x;

	// shouldn't happen because of launch spec but check anyways...
	if ((output_class >= output_dim) || (sample_ind >= batch_size)){
		return;
	}
	output_deriv[sample_ind * output_dim + output_class] /= batch_size;
}


// launch with gridDim = (batch_size), blockDim = (1)
__global__ void crossEntropyDeriv(float * output_deriv, const int * correct_classes, int output_dim, int batch_size){
	int i = threadIdx.x;
	if (i < batch_size){
		output_deriv[i * output_dim + correct_classes[i]] -= 1;
	}
}

// assume large 1-D launch
__global__ void updateMeans(int size, const float * gradients, const float * model_params, float base_mean_decay, float weight_decay, float * prev_means, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	float grad_with_decay = gradients[i] + weight_decay * model_params[i];
	prev_means[i] = base_mean_decay * prev_means[i] + (1 - base_mean_decay) * grad_with_decay;
	
}

// assume large 1-D launch
__global__ void updateVars(int size, const float * gradients, const float * model_params, float base_var_decay, float weight_decay, float * prev_vars, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	float grad_with_decay = gradients[i] + weight_decay * model_params[i];
	prev_vars[i] = base_var_decay * prev_vars[i] + (1 - base_var_decay) * grad_with_decay * grad_with_decay;
}

// assume large 1-D launch
__global__ void updateParams(int size, float * model_params, const float * means, const float * vars, float learning_rate, float weight_decay, float cur_mean_decay, float cur_var_decay, float eps, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	float mean_adj = means[i] / (1 - cur_mean_decay);
	float var_adj = vars[i] / (1 - cur_var_decay);
	float old_model_param = model_params[i];
	model_params[i] = model_params[i] - (learning_rate * (mean_adj / (sqrtf(var_adj) + eps)) + weight_decay * old_model_param);
}

/* INITIALIZE CORE MODEL STRUCTURES */

Dims * init_dimensions(int input, int init_kernel_dim, int init_conv_filters, int init_conv_stride, int init_maxpool_dim, int init_maxpool_stride, 
							int n_conv_blocks, int * is_block_spatial_reduction, int final_depth, int output){
	
	Dims * dims = (Dims *) malloc(sizeof(Dims));
	dims -> input = input;
	dims -> init_kernel_dim = init_kernel_dim;
	dims -> init_conv_filters = init_conv_filters;
	dims -> init_conv_stride = init_conv_stride;
	dims -> init_maxpool_dim = init_maxpool_dim;
	dims -> init_maxpool_stride = init_maxpool_stride;
	dims -> n_conv_blocks = n_conv_blocks;
	dims -> is_block_spatial_reduction = is_block_spatial_reduction;
	dims -> final_depth = final_depth;
	dims -> output = output;

	return dims;
}

BatchNorm * init_batch_norm(int spatial_dim, int depth, float gamma_val, bool is_zero){
	
	BatchNorm * batch_norm = (BatchNorm *) malloc(sizeof(BatchNorm));

	batch_norm -> spatial_dim = spatial_dim;
	batch_norm -> depth = depth;

	float * gamma, * beta;

	hipMalloc(&gamma, depth * sizeof(float));
	hipMemset(gamma, 0, depth * sizeof(float));
	// ZERO-GAMMA INITIALIZE TO SOLVE PROBLEM OF EXPLODING GRADIENTS (Goyal et al. 2017)
	if (!is_zero){
		setVal <<< ceil((float) depth / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (depth, gamma_val, gamma);
	}

	hipMalloc(&beta, depth * sizeof(float));
	hipMemset(beta, 0, depth * sizeof(float));

	batch_norm -> gamma = gamma;
	batch_norm -> beta = beta;

	return batch_norm;

}

ConvBlock * init_conv_block(int incoming_filters, int incoming_spatial_dim, int reduced_depth, int expanded_depth, int stride, hiprandGenerator_t * gen, bool is_zero){
	
	ConvBlock * conv_block = (ConvBlock *) malloc(sizeof(ConvBlock));
	conv_block -> incoming_filters = incoming_filters;
	conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	conv_block -> reduced_depth = reduced_depth;
	conv_block -> expanded_depth = expanded_depth;
	conv_block -> stride = stride;

	float * depth_reduction, *spatial, *depth_expansion;
	int depth_reduction_size, spatial_size, depth_expansion_size;
	float depth_reduction_fan_in_plus_fan_out, spatial_fan_in_plus_fan_out, depth_expansion_fan_in_plus_fan_out;

	BatchNorm *norm_depth_reduction, *norm_spatial, *norm_expansion, *norm_projection;

	depth_reduction_size = incoming_filters * reduced_depth;
	depth_reduction_fan_in_plus_fan_out = incoming_filters + reduced_depth;
	hipMalloc(&depth_reduction, depth_reduction_size * sizeof(float));
	hipMemset(depth_reduction, 0, depth_reduction_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, depth_reduction_size, depth_reduction, 0, 2.0 / depth_reduction_fan_in_plus_fan_out);
	}

	norm_depth_reduction = init_batch_norm(incoming_spatial_dim, reduced_depth, 1.0, is_zero);


	spatial_size = reduced_depth * reduced_depth * 3 * 3;
	spatial_fan_in_plus_fan_out = (3 * 3) * (reduced_depth + reduced_depth);
	hipMalloc(&spatial, spatial_size * sizeof(float));
	hipMemset(spatial, 0, spatial_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, spatial_size, spatial, 0, 2.0 / spatial_fan_in_plus_fan_out);
	}
	// the spatial decrease happens at middle 3x3 layer, to the last layer of stride block will receive lower spatial dim input
	if (stride == 2){
		incoming_spatial_dim /= 2;
	}
	norm_spatial = init_batch_norm(incoming_spatial_dim, reduced_depth, 1.0, is_zero);

	depth_expansion_size = expanded_depth * reduced_depth;
	depth_expansion_fan_in_plus_fan_out = reduced_depth + expanded_depth;
	hipMalloc(&depth_expansion, depth_expansion_size * sizeof(float));
	hipMemset(depth_expansion, 0, depth_expansion_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, depth_expansion_size, depth_expansion, 0, 2.0 / depth_expansion_fan_in_plus_fan_out);
	}
	conv_block -> depth_reduction = depth_reduction;
	conv_block -> norm_depth_reduction = norm_depth_reduction;

	conv_block -> spatial = spatial;
	conv_block -> norm_spatial = norm_spatial;


	conv_block -> depth_expansion = depth_expansion;

	norm_expansion = init_batch_norm(incoming_spatial_dim, expanded_depth, 1.0, is_zero);
	conv_block -> norm_expansion = norm_expansion;

	float * projection;
	int projection_size;
	if (stride == 2){
		projection_size = 3 * 3 * incoming_filters * expanded_depth;
	}
	else{
		projection_size = incoming_filters * expanded_depth;
	}

	// assuming only project when depths are different (all projections in resnet-50 this way)
	// could later change to adapt to just spatial transform...
	int projection_fan_in_plus_fan_out;
	if (incoming_filters != expanded_depth){
		hipMalloc(&projection, projection_size * sizeof(float));
		hipMemset(projection, 0, projection_size * sizeof(float));
		if (stride == 2){
			projection_fan_in_plus_fan_out = 3 * 3 * (incoming_filters + expanded_depth);
		}
		else{
			projection_fan_in_plus_fan_out = incoming_filters + expanded_depth;
		}
		if (!is_zero){
			init_weights_gaussian_device(gen, projection_size, projection, 0, 2.0 / (projection_fan_in_plus_fan_out));
		}
		norm_projection = init_batch_norm(incoming_spatial_dim, expanded_depth, 1.0, is_zero);
	}
	else{
		projection = NULL;
		norm_projection = NULL;
	}

	conv_block -> projection = projection;
	conv_block -> norm_projection = norm_projection;

	return conv_block;
}

Params * init_model_parameters(Dims * model_dims, hiprandGenerator_t * gen, bool is_zero){

	Params * params = (Params *) malloc(sizeof(Params));

	// dimensions unpacked
	int input_dim = model_dims -> input;
	int n_conv_blocks = model_dims -> n_conv_blocks;
	int init_kernel_dim = model_dims -> init_kernel_dim;
	int init_conv_filters = model_dims -> init_conv_filters;
	int * is_block_spatial_reduction = model_dims -> is_block_spatial_reduction;
	int output_dim = model_dims -> output;

	// init array to hold pointers to weights
	// 3 * 3 weight arrays per conv block (weights, gamma, beta per layer in block) + 3 * inital + fully connected + 4 projections * 3
	int n_locations = 16 + 9 * n_conv_blocks;
	params -> n_locations = n_locations;

	float ** locations = (float **) malloc(n_locations * sizeof(float *));
	int * sizes = (int *) malloc(n_locations * sizeof(int));
	// tracking location ind as we start allocating...
	


	// init first 7 * 7 conv_layer
	float * init_conv_layer;
	int init_conv_size = init_kernel_dim * init_kernel_dim * init_conv_filters * 3;
	float init_conv_fan_in_plus_fan_out = 7 * 7 * (3 + init_conv_filters);
	hipError_t malloc_err = hipMalloc(&init_conv_layer,  init_conv_size * sizeof(float));
	hipError_t memset_err = hipMemset(init_conv_layer, 0, init_conv_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, init_conv_size, init_conv_layer, 0, 2.0 / init_conv_fan_in_plus_fan_out);
	}
	params -> init_conv_layer = init_conv_layer;
	int loc_ind = 0;
	locations[loc_ind] = init_conv_layer;
	sizes[loc_ind] = init_kernel_dim * init_kernel_dim * init_conv_filters * 3;
	loc_ind++;

	BatchNorm * norm_init_conv = init_batch_norm(input_dim / model_dims -> init_conv_stride, init_conv_filters, 1.0, is_zero);
	params -> norm_init_conv = norm_init_conv;

	locations[loc_ind] = norm_init_conv -> gamma;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;

	locations[loc_ind] = norm_init_conv -> beta;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;
	

	// init conv blocks
	ConvBlock ** conv_blocks = (ConvBlock **) malloc(n_conv_blocks * sizeof(ConvBlock *));
	int incoming_filters = init_conv_filters;
	// assume stride 2 initial conv layer then stride 2 pool before entering conv_blocks
	int incoming_spatial_dim = input_dim / 4;
	int stride = 1;
	int reduced_depth = init_conv_filters;
	int expanded_depth = 4 * init_conv_filters;
	for (int i = 0; i < n_conv_blocks; i++){
		if (is_block_spatial_reduction[i] == 1){
			stride = 2;
			reduced_depth *= 2;
			expanded_depth *= 2;
		}
		else{
			stride = 1;
		}
		conv_blocks[i] = init_conv_block(incoming_filters, incoming_spatial_dim, reduced_depth, expanded_depth, stride, gen, is_zero);
		locations[loc_ind] = conv_blocks[i] -> depth_reduction;
		sizes[loc_ind] = incoming_filters * reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_depth_reduction -> gamma;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_depth_reduction -> beta;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> spatial;
		sizes[loc_ind] = reduced_depth * reduced_depth * 3 * 3;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_spatial -> gamma;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_spatial -> beta;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> depth_expansion;
		sizes[loc_ind] = expanded_depth * reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> norm_expansion -> gamma;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_expansion -> beta;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;
		
		// if the block needed a projection to make input dim = output dim
		if (conv_blocks[i] -> projection){
			locations[loc_ind] = conv_blocks[i] -> projection;
			if (stride == 2){
				sizes[loc_ind] = 3 * 3 * incoming_filters * expanded_depth;
			}
			else{
				sizes[loc_ind] = incoming_filters * expanded_depth;
			}
			loc_ind++;
			locations[loc_ind] = conv_blocks[i] -> norm_projection -> gamma;
			sizes[loc_ind] = expanded_depth;
			loc_ind++;
			locations[loc_ind] = conv_blocks[i] -> norm_projection -> beta;
			sizes[loc_ind] = expanded_depth;
			loc_ind++;
		}

		// after stride 2 block then reduce spatial dim for next block
		if (is_block_spatial_reduction[i] == 1){
			incoming_spatial_dim /= 2;
		}
		incoming_filters = expanded_depth;
	}
	params -> conv_blocks = conv_blocks;

	float * fully_connected;
	// here expanded depth is the last layer's filters which will go through average pool before FC layer
	// expanded depth should equal dims -> final_depth
	int fully_connected_size = expanded_depth * output_dim;
	float fully_connected_fan_in = expanded_depth;
	hipMalloc(&fully_connected, fully_connected_size * sizeof(float));
	hipMemset(fully_connected, 0, fully_connected_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, fully_connected_size, fully_connected, 0, 0.0001);
	}

	params -> fully_connected = fully_connected;
	locations[loc_ind] = fully_connected;
	sizes[loc_ind] = expanded_depth * output_dim;

	params -> locations = locations;
	params -> sizes = sizes;

	return params;
}

ResNet * init_resnet(Dims * dims, hiprandGenerator_t * gen){
	ResNet * model = (ResNet *) malloc(sizeof(ResNet));
	model -> dims = dims;
	Params * params = init_model_parameters(dims, gen, false);
	model -> params = params;
	return model;
}


/* INITIALIZE TRAINING STRUCTURES */

Cache_BatchNorm * init_cache_batchnorm(int input_size, int feature_size){
	Cache_BatchNorm * cache_batchnorm = (Cache_BatchNorm *) malloc(sizeof(Cache_BatchNorm));

	cache_batchnorm -> input_size = input_size;
	cache_batchnorm -> feature_size = feature_size;

	float * means, *inv_vars;
	hipMalloc(&means, feature_size * sizeof(float));
	hipMemset(means, 0, feature_size * sizeof(float));
	hipMalloc(&inv_vars, feature_size * sizeof(float));
	hipMemset(inv_vars, 0, feature_size * sizeof(float));

	cache_batchnorm -> means = means;
	cache_batchnorm -> inv_vars = inv_vars;

	return cache_batchnorm;
}

Activation_ConvBlock * init_activation_convblock(ConvBlock * conv_block, int batch_size, size_t * max_activation_size){
	Activation_ConvBlock * activation_conv_block = (Activation_ConvBlock *) malloc(sizeof(Activation_ConvBlock));

	int incoming_filters = conv_block -> incoming_filters;
	int incoming_spatial_dim = conv_block -> incoming_spatial_dim;
	int stride = conv_block -> stride;
	int reduced_depth = conv_block -> reduced_depth;
	int expanded_depth = conv_block -> expanded_depth;

	activation_conv_block -> incoming_filters = incoming_filters;
	activation_conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	activation_conv_block -> reduced_depth = reduced_depth;
	activation_conv_block -> expanded_depth = expanded_depth;
	activation_conv_block -> stride = stride;

	float * post_reduced, *post_spatial, *post_expanded, *transformed_residual, *post_projection_norm_vals, *output;
	float * post_reduced_activated, *post_spatial_activated;
	int post_reduced_size, post_spatial_size, output_size;
	Cache_BatchNorm * norm_post_reduced, *norm_post_spatial, *norm_post_expanded, *norm_post_projection;
	

	post_reduced_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim * batch_size;
	hipMalloc(&post_reduced, post_reduced_size * sizeof(float));
	activation_conv_block -> post_reduced = post_reduced;

	norm_post_reduced = init_cache_batchnorm(post_reduced_size, reduced_depth);
	activation_conv_block -> norm_post_reduced = norm_post_reduced;

	hipMalloc(&post_reduced_activated, post_reduced_size * sizeof(float));
	activation_conv_block -> post_reduced_activated = post_reduced_activated;

	if (*max_activation_size < post_reduced_size){
		*max_activation_size = post_reduced_size;
	}

	post_spatial_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim / (stride * stride) * batch_size;
	hipMalloc(&post_spatial, post_spatial_size * sizeof(float));
	activation_conv_block -> post_spatial = post_spatial;

	norm_post_spatial = init_cache_batchnorm(post_spatial_size, reduced_depth);
	activation_conv_block -> norm_post_spatial = norm_post_spatial;

	hipMalloc(&post_spatial_activated, post_spatial_size * sizeof(float));
	activation_conv_block -> post_spatial_activated = post_spatial_activated;

	if (*max_activation_size < post_spatial_size){
		*max_activation_size = post_spatial_size;
	}

	output_size = expanded_depth * incoming_spatial_dim * incoming_spatial_dim / (stride * stride) * batch_size;
	
	hipMalloc(&post_expanded, output_size * sizeof(float));
	activation_conv_block -> post_expanded = post_expanded;

	norm_post_expanded = init_cache_batchnorm(output_size, expanded_depth);
	activation_conv_block -> norm_post_expanded = norm_post_expanded;

	// only allocate space if transformed, otherwise it will be assumed to be identity of input
	transformed_residual = NULL;
	norm_post_projection = NULL;
	post_projection_norm_vals = NULL;
	if (incoming_filters != expanded_depth){
		hipMalloc(&transformed_residual, output_size * sizeof(float));
		norm_post_projection = init_cache_batchnorm(output_size, expanded_depth);
		hipMalloc(&post_projection_norm_vals, output_size * sizeof(float));
	}
	activation_conv_block -> transformed_residual = transformed_residual;
	activation_conv_block -> norm_post_projection = norm_post_projection;
	activation_conv_block -> post_projection_norm_vals = post_projection_norm_vals;

	hipMalloc(&output, output_size * sizeof(float));
	activation_conv_block -> output = output;

	if (*max_activation_size < output_size){
		*max_activation_size = output_size;
	}

	return activation_conv_block;
}

Activations * init_activations(Dims * dims, ConvBlock ** conv_blocks, int batch_size){
	
	Activations * activations = (Activations *) malloc(sizeof(Activations));

	int input_dim = dims -> input;
	int init_conv_filters = dims -> init_conv_filters;
	int init_conv_stride = dims -> init_conv_stride;
	int maxpool_stride = dims -> init_maxpool_stride;


	float * init_conv_applied;
	int init_conv_applied_size = init_conv_filters * input_dim * input_dim / (init_conv_stride * init_conv_stride) * batch_size; 
	hipMalloc(&init_conv_applied, init_conv_applied_size * sizeof(float));
	activations -> init_conv_applied = init_conv_applied;

	size_t max_activation_size = init_conv_applied_size;

	Cache_BatchNorm * norm_init_conv = init_cache_batchnorm(init_conv_applied_size, init_conv_filters);
	activations -> norm_init_conv = norm_init_conv;

	float * init_conv_activated;
	hipMalloc(&init_conv_activated, init_conv_applied_size * sizeof(float));
	activations -> init_conv_activated = init_conv_activated;

	int init_convblock_input_size = init_conv_filters * input_dim * input_dim / (init_conv_stride * init_conv_stride) / (maxpool_stride * maxpool_stride) * batch_size;

	float *init_convblock_input;
	hipMalloc(&init_convblock_input, init_convblock_input_size * sizeof(float));
	activations -> init_convblock_input = init_convblock_input;

	if (init_convblock_input_size > max_activation_size){
		max_activation_size = init_convblock_input_size;
	}

	int n_conv_blocks = dims -> n_conv_blocks;

	Activation_ConvBlock ** activation_conv_blocks = (Activation_ConvBlock **) malloc(n_conv_blocks * sizeof(Activation_ConvBlock *));
	for (int i = 0; i < n_conv_blocks; i++){
		ConvBlock * conv_block = conv_blocks[i];
		activation_conv_blocks[i] = init_activation_convblock(conv_block, batch_size, &max_activation_size);
	}

	activations -> activation_conv_blocks = activation_conv_blocks;
	activations -> n_conv_blocks = n_conv_blocks;

	int final_depth = dims -> final_depth;
	float * final_conv_output_pooled;
	int final_conv_output_pooled_size = final_depth * batch_size;
	hipMalloc(&final_conv_output_pooled, final_conv_output_pooled_size * sizeof(float));
	activations -> final_conv_output_pooled = final_conv_output_pooled;

	if (final_conv_output_pooled_size > max_activation_size){
		max_activation_size = final_conv_output_pooled_size;
	}

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * linear_output;
	hipMalloc(&linear_output, output_size * sizeof(float));
	activations -> linear_output = linear_output;

	activations -> max_activation_size = max_activation_size;

	return activations;
}


Forward_Buffer * init_forward_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Forward_Buffer * forward_buffer = (Forward_Buffer *) malloc(sizeof(Forward_Buffer));

	forward_buffer -> activations = init_activations(dims, conv_blocks, batch_size);

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * pred;
	hipMalloc(&pred, output_size * batch_size * sizeof(float));
	forward_buffer -> pred = pred;

	// will be copied to cpu to be able to print values and compute loss on cpu side
	float * pred_cpu = (float *) malloc(output_size * batch_size * sizeof(float));
	forward_buffer -> pred_cpu = pred_cpu;

	// init + 3 * n_conv_blocks + 3 residual projections
	int * forward_conv_algos = (int *) malloc(53 * sizeof(int));
	size_t * forward_workspace = (size_t *) malloc(53 * sizeof(size_t));
	for (int i = 0; i < 53; i++){
		forward_conv_algos[i] = -1;
	}

	forward_buffer -> conv_algos = forward_conv_algos;
	forward_buffer -> workspace_algos = forward_workspace;
	return forward_buffer;
}


Backprop_Buffer * init_backprop_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Backprop_Buffer * backprop_buffer = (Backprop_Buffer *) malloc(sizeof(Backprop_Buffer));

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * output_layer_deriv;
	hipMalloc(&output_layer_deriv, output_size * sizeof(float));
	backprop_buffer -> output_layer_deriv = output_layer_deriv;

	backprop_buffer -> param_derivs = init_model_parameters(dims, NULL, true);
	backprop_buffer -> prev_means = init_model_parameters(dims, NULL, true);
	backprop_buffer -> prev_vars = init_model_parameters(dims, NULL, true);

	int * data_algos = (int *) malloc(53 * sizeof(int));
	int * filt_algos = (int *) malloc(53 * sizeof(int));
	for (int i = 0; i < 53; i++){
		data_algos[i] = -1;
		filt_algos[i] = -1;
	}

	size_t * workspace_data = (size_t *) malloc(53 * sizeof(size_t));
	size_t * workspace_filt = (size_t *) malloc(53 * sizeof(size_t));

	backprop_buffer -> data_algos = data_algos;
	backprop_buffer -> filt_algos = filt_algos;
	backprop_buffer -> workspace_data = workspace_data;
	backprop_buffer -> workspace_filt = workspace_filt;


	return backprop_buffer;
}


Train_ResNet * init_trainer(int id, ResNet * model, Batch * cur_batch, Shared_Struct * shared_struct, int batch_size, float learning_rate, float weight_decay, float mean_decay, float var_decay, float eps, int n_epochs, miopenHandle_t * handle, const char * dump_dir){
	Train_ResNet * trainer = (Train_ResNet *) malloc(sizeof(Train_ResNet));

	trainer -> id = id;

	trainer -> model = model;

	trainer -> cur_batch = cur_batch;
	trainer -> batch_size = batch_size;

	Dims * dims = model -> dims;
	ConvBlock ** conv_blocks = model -> params -> conv_blocks;
	trainer -> forward_buffer = init_forward_buffer(dims, conv_blocks, batch_size);
	trainer -> backprop_buffer = init_backprop_buffer(dims, conv_blocks, batch_size);
	trainer -> shared_struct = shared_struct;

	trainer -> learning_rate = learning_rate;
	trainer -> weight_decay = weight_decay;
	trainer -> base_mean_decay = mean_decay;
	trainer -> base_var_decay = var_decay;
	
	trainer -> cur_mean_decay = 1;
	trainer -> cur_var_decay = 1;
	
	trainer -> eps = eps;

	trainer -> n_epochs = n_epochs;

	trainer -> cur_dump_id = -1;

	trainer -> cur_epoch = 0;

	trainer -> loss_per_epoch = (float *) calloc(n_epochs, sizeof(float));
	trainer -> accuracy_per_epoch = (float *) calloc(n_epochs, sizeof(float));

	trainer -> init_loaded = 0;

	trainer -> miopenHandle = *handle;

	trainer -> dump_dir = dump_dir;

	return trainer;
}

Batch * init_general_batch(int n_images, int image_size, int image_dim, int shard_n_images){
	Batch * batch = (Batch *) malloc(sizeof(Batch));

	batch -> n_images = n_images;
	// in resnet-50 will be 224 * 224 * 3
	batch -> image_size = image_size;
	batch -> image_dim = image_dim;
	float * images_float_cpu;
	// load batch by first brining into cpu, pinned memory
	hipError_t status_images_pinned = hipMalloc(&images_float_cpu, (size_t) n_images * (size_t) image_size * sizeof(float));
	batch -> images_float_cpu = images_float_cpu;
	
	// allocate memory on gpu so that after loaded on cpu can bring in
	// will be converting from uint8 on CPU to float on GPU
	float * images;
	hipMalloc(&images, (size_t) n_images * (size_t) image_size * sizeof(float));
	batch -> images = images;

	// pinned memory for correct_classes_cpu
	int * correct_classes_cpu;
	hipError_t status_classes_pinned = hipMalloc(&correct_classes_cpu, n_images * sizeof(int));
	batch -> correct_classes_cpu = correct_classes_cpu;

	int * correct_classes;
	hipMalloc(&correct_classes, n_images * sizeof(int));
	batch -> correct_classes = correct_classes;

	batch -> cur_shard_id = -1;
	batch -> cur_batch_in_shard = -1;
	
	batch -> shard_n_images = shard_n_images;
	batch -> full_shard_images = (float *) malloc((size_t) shard_n_images * (size_t) image_size * sizeof(float));
	batch -> full_shard_correct_classes = (int *) malloc(shard_n_images * sizeof(int));

	return batch;
}

// (if this takes too long, can do it in parallel with separate process on cpu)
// ASSUMING shard_n_images % batch_size = 0
void load_new_batch(Train_ResNet * trainer, Class_Metadata * class_metadata, Batch * batch_buffer){
	
	int batch_size = batch_buffer -> n_images;
	int image_size = batch_buffer -> image_size;
	int image_dim = batch_buffer -> image_dim;
	
	
	float * full_shard_images = batch_buffer -> full_shard_images;
	int * full_shard_correct_classes = batch_buffer -> full_shard_correct_classes;	

	float * images_float_cpu = batch_buffer -> images_float_cpu;
	float * images = batch_buffer -> images;

	int * correct_classes_cpu = batch_buffer -> correct_classes_cpu;
	int * correct_classes = batch_buffer -> correct_classes;

	int cur_shard_id = batch_buffer -> cur_shard_id;
	int cur_batch_in_shard = batch_buffer -> cur_batch_in_shard;
	int shard_n_images = batch_buffer -> shard_n_images;

	int cur_dump_id = trainer -> cur_dump_id;

	int init_loaded = trainer -> init_loaded;



	int start_img_num = cur_batch_in_shard * batch_size;

	// skip some images if shard_n_images not divisible by batch size
	if (start_img_num + batch_size > shard_n_images){
		start_img_num = shard_n_images;
	}

	int n_read;
	int print_ret;

	size_t total_pixels = (size_t) batch_size * (size_t) image_size;

	char * shard_images_filepath, * shard_labels_filepath;
	// cur_shard_id = -1 implies first iteration
	if ((init_loaded) || (cur_shard_id == -1) || (start_img_num >= shard_n_images)) {

		// update new shard id if first iter or passed the bounds
		if (! init_loaded){
			cur_shard_id += 1;
			batch_buffer -> cur_shard_id = cur_shard_id;
		}

		// load new shard into RAM
		print_ret = asprintf(&shard_images_filepath, "../sample_data/%03d.images", cur_shard_id);
		FILE * shard_images_file = fopen(shard_images_filepath, "rb");
		n_read = fread(full_shard_images, sizeof(float), ((size_t) shard_n_images) * ((size_t) image_size), shard_images_file);
		fclose(shard_images_file);
		free(shard_images_filepath);

		print_ret = asprintf(&shard_labels_filepath, "../sample_data/%03d.labels", cur_shard_id);
		FILE * shard_labels_file = fopen(shard_labels_filepath, "rb");
		n_read = fread(full_shard_correct_classes, sizeof(int), shard_n_images, shard_labels_file);
		fclose(shard_labels_file);
		free(shard_labels_filepath);

		// reset cur batch in shard to 0 if first iter or passed the bounds
		if (! init_loaded) {
			cur_batch_in_shard = 0;
			batch_buffer -> cur_batch_in_shard = cur_batch_in_shard;
		}

		// don't have to load special first batch from checkpoint anymore
		trainer -> init_loaded = 0;
	}

	// load current batch
	memcpy(images_float_cpu, full_shard_images + cur_batch_in_shard * total_pixels, total_pixels * sizeof(float));
	memcpy(correct_classes_cpu, full_shard_correct_classes + cur_batch_in_shard * batch_size, batch_size * sizeof(int));
	

	/* SAVING BATCH TO FILES FOR INSPECTION... */
	// if (cur_batch_in_shard == 0){
	// 	FILE * test_images_file = fopen("images.buffer", "wb");
	// 	fwrite(images_float_cpu, sizeof(float), total_pixels, test_images_file);
	// 	fclose(test_images_file);

	// 	FILE * test_labels_file = fopen("labels.buffer", "wb");
	// 	fwrite(correct_classes_cpu, sizeof(int), (size_t) batch_size, test_labels_file);
	// 	fclose(test_labels_file);
	// 	exit(0);
	// }

	// copy current batch to GPU

	hipMemcpy(images, images_float_cpu, total_pixels * sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(correct_classes, correct_classes_cpu, batch_size * sizeof(int), hipMemcpyHostToDevice);

	// update cur batch for next iteration of loading
	cur_batch_in_shard++;
	batch_buffer -> cur_batch_in_shard = cur_batch_in_shard;

	cur_dump_id++;
	trainer -> cur_dump_id = cur_dump_id;

}


// READ CLASSES AND LABELS!
// reading a text file line by line into a buffer
// pre-allocate buffer and specify type
void text_file_to_buffer(void * buffer, char * filename, const char * type){

	char ** my_text_buffer = (char **) buffer;
	int * my_int_buffer = (int *) buffer;
	
	FILE * fp;
    char * line = NULL;
    size_t len = 0;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int cnt = 0;
    while (getline(&line, &len, fp) != -1) {
    	if (strcmp(type, "TEXT") == 0){
        	my_text_buffer[cnt] = strdup(line);
        }
        else if (strcmp(type, "INT") == 0){
        	my_int_buffer[cnt] = atoi(line);
        }
        else{
        	// pass
        }
        cnt++;
    }

    fclose(fp);
    if (line){
    	free(line);
    }
}

Class_Metadata * populate_class_info(char * label_filename, char * synset_filename, char * class_size_filename, int n_classes){
	
	Class_Metadata * classes = (Class_Metadata *) malloc(sizeof(Class_Metadata));

	char ** labels = (char **) malloc(n_classes * sizeof(char *));
	char ** synsets = (char **) malloc(n_classes * sizeof(char *));
	int * counts = (int *) malloc(n_classes * sizeof(int));

	text_file_to_buffer(labels, label_filename, "TEXT");
	text_file_to_buffer(synsets, synset_filename, "TEXT");
	text_file_to_buffer(counts, class_size_filename, "INT");

	classes -> labels = labels;
	classes -> synsets = synsets;
	classes -> counts = counts;
	classes -> n_classes = n_classes;

	return classes;
}

/* PREP AND LAUNCHING CUDA KERNELS! */

// assume NCHW packed with each tensor having size
void prepareAndDoTensorOp(Train_ResNet * trainer, char * op_type, size_t size, int spatial_dim, int filters, int batch_size, float * A, float * B, float * C){
	miopenStatus_t status;

	miopenTensorDescriptor_t tensor_descriptor;
	status = miopenCreateTensorDescriptor(&tensor_descriptor);
	status = miopenSet4dTensorDescriptor(tensor_descriptor, miopenFloat, batch_size, filters, spatial_dim, spatial_dim);


	float alpha1 = 1, alpha2 = 1, beta = 0;
	miopenTensorOp_t tensor_op;
	if (strcmp(op_type, "ADD") == 0){
		tensor_op = miopenTensorOpAdd;
	}
	else if (strcmp(op_type, "SUB") == 0){
		tensor_op = miopenTensorOpAdd;
		alpha2 = -1;
	}
	else if (strcmp(op_type, "MUL") == 0){
		tensor_op = miopenTensorOpMul;
	}
	else if (strcmp(op_type, "MIN") == 0){
		tensor_op = miopenTensorOpMin;
	}
	else if (strcmp(op_type, "MAX") == 0){
		tensor_op = miopenTensorOpMax;
	}
	else{

		// ERROR!
	}

	status = miopenOpTensor(trainer -> miopenHandle, tensor_op, &alpha1, tensor_descriptor, A, &alpha2, tensor_descriptor, B, &beta, tensor_descriptor, C);

	miopenDestroyTensorDescriptor(tensor_descriptor);

}



void prepareAndDoActivation(Train_ResNet * trainer, size_t size, float * input, int spatial_dim, int filters, int batch_size, float *output){
	miopenStatus_t status;

	miopenTensorDescriptor_t input_descriptor;
	status = miopenCreateTensorDescriptor(&input_descriptor);
	status = miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, filters, spatial_dim, spatial_dim);

	miopenActivationDescriptor_t activation_descriptor;
	miopenCreateActivationDescriptor(&activation_descriptor);

	miopenSetActivationDescriptor(activation_descriptor, miopenActivationRELU, 1, 1, 1);

	const float alpha = 1.0, beta = 0;

	status = miopenActivationForward(trainer -> miopenHandle, activation_descriptor, &alpha, input_descriptor, input, &beta, input_descriptor, output);

	if (status != 0){
		printf("Cudnn status after forward activ: %s\n\n", miopenGetErrorString(status));
	}

	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyActivationDescriptor(activation_descriptor);

}

void prepareAndDoActivationDeriv(Train_ResNet * trainer, size_t size, float * input, float * out_layer_deriv, float * output, int spatial_dim, int filters, int batch_size, float *input_deriv){
	miopenStatus_t status;

	miopenTensorDescriptor_t tensor_descriptor;
	status = miopenCreateTensorDescriptor(&tensor_descriptor);
	status = miopenSet4dTensorDescriptor(tensor_descriptor, miopenFloat, batch_size, filters, spatial_dim, spatial_dim);

	miopenActivationDescriptor_t activation_descriptor;
	miopenCreateActivationDescriptor(&activation_descriptor);

	miopenSetActivationDescriptor(activation_descriptor, miopenActivationRELU, 1, 1, 1);

	const float alpha = 1.0, beta = 0;

	miopenActivationBackward(trainer -> miopenHandle, activation_descriptor, &alpha, tensor_descriptor, output, tensor_descriptor, out_layer_deriv, tensor_descriptor, input, &beta, tensor_descriptor, input_deriv);

	if (status != 0){
		printf("Cudnn status after backward activ: %s\n\n", miopenGetErrorString(status));
	}

	miopenDestroyTensorDescriptor(tensor_descriptor);
	miopenDestroyActivationDescriptor(activation_descriptor);

}

void prepareAndDoPool(Train_ResNet * trainer, const char * poolType, const float * input, int in_spatial_dim, int filters, int kern_dim, int stride, int batch_size, float * out){
	miopenStatus_t status;

	int out_spatial_dim = in_spatial_dim / stride;

	miopenTensorDescriptor_t input_descriptor;
	status = miopenCreateTensorDescriptor(&input_descriptor);
	status = miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, filters, in_spatial_dim, in_spatial_dim);

	

	miopenPoolingDescriptor_t pool_descriptor;
	miopenCreatePoolingDescriptor(&pool_descriptor);

	miopenPoolingMode_t poolMode = miopenPoolingMax;

	int pad = kern_dim / 2;
	if (strcmp(poolType, "AVG") == 0){
		poolMode = miopenPoolingAverage;
		pad = 0;
	}
	miopenSet2dPoolingDescriptor(pool_descriptor, poolMode, kern_dim, kern_dim, pad, pad, stride, stride);

	const float alpha = 1, beta = 0;

	int n, c, h, w;
	miopenGetPoolingForwardOutputDim(pool_descriptor, input_descriptor, &n, &c, &h, &w);

	miopenTensorDescriptor_t out_descriptor;
	status = miopenCreateTensorDescriptor(&out_descriptor);
	status = miopenSet4dTensorDescriptor(out_descriptor, miopenFloat, n, c, h, w);

	status = miopenPoolingForward(trainer -> miopenHandle, pool_descriptor, &alpha, input_descriptor, input, &beta, out_descriptor, out, false, NULL, 0);

	if (status != 0){
		printf("Cudnn status after forward max pool: %s\n\n", miopenGetErrorString(status));
	}

	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyTensorDescriptor(out_descriptor);
	miopenDestroyPoolingDescriptor(pool_descriptor);
}

void prepareAndDoPoolDeriv(Train_ResNet * trainer, const char * poolType, const float * input, float * output, float * out_layer_deriv, int in_spatial_dim, int filters, int kern_dim, int stride, int batch_size, float * input_deriv){
	miopenStatus_t status;

	int out_spatial_dim = in_spatial_dim / stride;

	miopenTensorDescriptor_t input_descriptor;
	status = miopenCreateTensorDescriptor(&input_descriptor);
	status = miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, filters, in_spatial_dim, in_spatial_dim);

	miopenTensorDescriptor_t out_descriptor;
	status = miopenCreateTensorDescriptor(&out_descriptor);
	status = miopenSet4dTensorDescriptor(out_descriptor, miopenFloat, batch_size, filters, out_spatial_dim, out_spatial_dim);

	miopenPoolingDescriptor_t pool_descriptor;
	miopenCreatePoolingDescriptor(&pool_descriptor);

	miopenPoolingMode_t poolMode = miopenPoolingMax;

	int pad = kern_dim / 2;
	if (strcmp(poolType, "AVG") == 0){
		poolMode = miopenPoolingAverage;
		pad = 0;
	}

	miopenSet2dPoolingDescriptor(pool_descriptor, poolMode, kern_dim, kern_dim, pad, pad, stride, stride);

	const float alpha = 1, beta = 0;

	void * workspace = NULL;
	size_t workspace_bytes = 0;
	miopenPoolingGetWorkSpaceSizeV2(pool_descriptor, out_descriptor, &workspace_bytes);

	hipMalloc(&workspace, workspace_bytes);
	status = miopenPoolingBackward(trainer -> miopenHandle, pool_descriptor, &alpha, out_descriptor, output, out_descriptor, out_layer_deriv, input_descriptor, input, &beta, input_descriptor, input_deriv, workspace);

	if (status != 0){
		printf("Cudnn status after backward max pool: %s\n\n", miopenGetErrorString(status));
	}

	hipFree(workspace);
	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyTensorDescriptor(out_descriptor);
	miopenDestroyPoolingDescriptor(pool_descriptor);

}

void prepareAndDoConvolution(Train_ResNet * trainer, int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output, int * algo, size_t * mem){
	miopenStatus_t status;

	miopenTensorDescriptor_t input_descriptor;
	status = miopenCreateTensorDescriptor(&input_descriptor);
	status = miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, in_filters, in_spatial_dim, in_spatial_dim);

	miopenTensorDescriptor_t kernel_descriptor;
	status = miopenCreateTensorDescriptor(&kernel_descriptor);
	status = miopenSet4dTensorDescriptor(kernel_descriptor, miopenFloat, out_filters, in_filters, kern_dim, kern_dim);

	miopenConvolutionDescriptor_t convolution_descriptor;
	status = miopenCreateConvolutionDescriptor(&convolution_descriptor);
	status = miopenInitConvolutionDescriptor(convolution_descriptor, miopenConvolution, kern_dim / 2, kern_dim / 2, stride, stride, 1, 1);

	int out_spatial_dim = in_spatial_dim / stride;

	miopenTensorDescriptor_t output_descriptor;
	status = miopenCreateTensorDescriptor(&output_descriptor);
	status = miopenSet4dTensorDescriptor(output_descriptor, miopenFloat, batch_size, out_filters, out_spatial_dim, out_spatial_dim);

	float total_time;
	float min_time = 1000000;
	int min_algo = 0;

	void * workspace = NULL;
	size_t workspace_bytes = 0;
	size_t min_mem = 0;

	

	if (*algo == -1){

		int returned_cnt;

		miopenConvolutionForwardGetWorkSpaceSize(trainer -> miopenHandle, kernel_descriptor, input_descriptor, convolution_descriptor, output_descriptor, &workspace_bytes);
		hipMalloc(&workspace, workspace_bytes);

		miopenConvAlgoPerf_t * perfResults = (miopenConvAlgoPerf_t *) malloc(MAX_REQ_CONV_FIND_ALGO * sizeof(miopenConvAlgoPerf_t));
		miopenFindConvolutionForwardAlgorithm(trainer -> miopenHandle, input_descriptor, input, kernel_descriptor, weights, convolution_descriptor, output_descriptor, output, MAX_REQ_CONV_FIND_ALGO, &returned_cnt, perfResults, workspace, workspace_bytes, true);

		hipFree(workspace);

		miopenConvAlgoPerf_t cur_perf_results;
		//FILE * fp_perf = fopen("kernel_perfs/forward_conv.text", "a");
		//fprintf(fp_perf, "\nConvolution Arguments: %d,%d,%d,%d,%d,%d\n", in_spatial_dim, kern_dim, stride, in_filters, out_filters, batch_size);
		for (int i = 0; i < returned_cnt; i++){
			cur_perf_results = perfResults[i];
			//fprintf(fp_perf, "%d,%f,%zu,%d\n", cur_perf_results.algo, cur_perf_results.time, cur_perf_results.memory, cur_perf_results.mathType);
			if (cur_perf_results.time > 0){
					// measured bandthwidth ratio of hipMalloc as MiB / 34 = ms delay
					//total_time = cur_perf_results.time + ((float) cur_perf_results.memory / 1e6) / 34;
					total_time = cur_perf_results.time;
					if (total_time < min_time){
						min_time = total_time;
						min_algo = cur_perf_results.fwd_algo;
						min_mem = cur_perf_results.memory;
					}
			}
		}
		//fclose(fp_perf);
		//free(perfResults);

		*algo = min_algo;
		*mem = min_mem;
		free(perfResults);
	
		// const algo_t algos[] = {
	    //       miopen_CONVOLUTION_FWD_ALGO_GEMM,
	    //       miopen_CONVOLUTION_FWD_ALGO_FFT,
	    //       miopen_CONVOLUTION_FWD_ALGO_FFT_TILING,
	    //       miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	    //       miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
	    //       miopen_CONVOLUTION_FWD_ALGO_DIRECT,
	    //       miopen_CONVOLUTION_FWD_ALGO_WINOGRAD,
	    //       miopen_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
	    //  };
	}


	miopenConvFwdAlgorithm_t convolution_algorithm = (miopenConvFwdAlgorithm_t) *algo;

	//printf("Forward Workspace Bytes: %zu\n", workspace_bytes);

	hipMalloc(&workspace, *mem);

	const float alpha = 1, beta = 0;
	status = miopenConvolutionForward(trainer -> miopenHandle, &alpha, input_descriptor, input, kernel_descriptor, weights, convolution_descriptor, convolution_algorithm, &beta, output_descriptor, output, workspace, *mem);

	hipFree(workspace);
	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyTensorDescriptor(output_descriptor);
	miopenDestroyTensorDescriptor(kernel_descriptor);
	miopenDestroyConvolutionDescriptor(convolution_descriptor);
}

void prepreAndDoConvolutionDeriv(Train_ResNet * trainer, int in_spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
												float * input, float * weights, float * out_deriv,
												float * input_deriv, float * weight_deriv, bool toComputeInputDeriv, int * dataAlgo, int *filtAlgo, size_t * memData, size_t * memFilt){

	int out_spatial_dim = in_spatial_dim / stride;

	miopenStatus_t status;

	miopenTensorDescriptor_t input_descriptor;
	miopenCreateTensorDescriptor(&input_descriptor);
	miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, in_filters, in_spatial_dim, in_spatial_dim);

	miopenTensorDescriptor_t output_descriptor;
	miopenCreateTensorDescriptor(&output_descriptor);
	miopenSet4dTensorDescriptor(output_descriptor, miopenFloat, batch_size, out_filters, out_spatial_dim, out_spatial_dim);

	miopenTensorDescriptor_t kernel_descriptor;
	miopenCreateTensorDescriptor(&kernel_descriptor);
	miopenSet4dTensorDescriptor(kernel_descriptor, miopenFloat, out_filters, in_filters, kern_dim, kern_dim);

	miopenConvolutionDescriptor_t convolution_descriptor;
	miopenCreateConvolutionDescriptor(&convolution_descriptor);
	miopenInitConvolutionDescriptor(convolution_descriptor, miopenConvolution, kern_dim / 2, kern_dim / 2, stride, stride, 1, 1);

	size_t inp_size = in_spatial_dim * in_spatial_dim * batch_size * in_filters;
	const float a_dummy = 1, b_dummy = 0;

	float alpha = 1, beta = 0;

	int returned_cnt;

	void * workspace = NULL;
	size_t workspace_bytes = 0;

	float total_time;
	float min_time = 1000000;
	int min_algo = 0;
	size_t min_mem = 0;

	// Compute deriv w.r.t input data
	if (toComputeInputDeriv){

		 // static const algo_t algos[] = {
         // miopen_CONVOLUTION_BWD_DATA_ALGO_0,
         // miopen_CONVOLUTION_BWD_DATA_ALGO_1,
         // miopen_CONVOLUTION_BWD_DATA_ALGO_FFT,
         // miopen_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
         // miopen_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
         // miopen_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
     	 // };

		//miopenConvAlgoPerf_t top_data_algo[1];
		//cudnnGetConvolutionBackwardDataAlgorithm_v7(trainer -> miopenHandle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor, 1, &returned_cnt, top_data_algo);
		
		

		if (*dataAlgo == -1){
			int returned_cnt;

			miopenConvolutionBackwardDataGetWorkSpaceSize(trainer -> miopenHandle, output_descriptor, kernel_descriptor, convolution_descriptor, input_descriptor, &workspace_bytes);
			hipMalloc(&workspace, workspace_bytes);

			miopenConvAlgoPerf_t * perfResultsData = (miopenConvAlgoPerf_t *) malloc(MAX_REQ_CONV_FIND_ALGO * sizeof(miopenConvAlgoPerf_t));
			miopenFindConvolutionBackwardDataAlgorithm(trainer -> miopenHandle, output_descriptor, out_deriv, kernel_descriptor, weights, convolution_descriptor, input_descriptor, input_deriv, MAX_REQ_CONV_FIND_ALGO, &returned_cnt, perfResultsData, workspace, workspace_bytes, true);

			hipFree(workspace);

			miopenConvAlgoPerf_t cur_perf_results;
			//FILE * fp_perf_data = fopen("kernel_perfs/backward_data_conv.text", "a");
			//fprintf(fp_perf_data, "\nConvolution Arguments: %d,%d,%d,%d,%d,%d\n", in_spatial_dim, kern_dim, stride, in_filters, out_filters, batch_size);
			for (int i = 0; i < returned_cnt; i++){
				cur_perf_results = perfResultsData[i];
				if (cur_perf_results.time > 0){
					// measured bandthwidth ratio of hipMalloc as MiB / 34 = ms delay
					//total_time = cur_perf_results.time + ((float) cur_perf_results.memory / 1e6) / 34;
					total_time = cur_perf_results.time;
					if (total_time < min_time){
						min_time = total_time;
						min_algo = cur_perf_results.bwd_data_algo;
						min_mem = cur_perf_results.memory;
					}
				}
				
				//fprintf(fp_perf_data, "%d,%f,%zu,%d\n", cur_perf_results.algo, cur_perf_results.time, cur_perf_results.memory, cur_perf_results.mathType);
			}
			//fclose(fp_perf_data);
			//free(perfResultsData);


			*dataAlgo = min_algo;
			*memData = min_mem;
			free(perfResultsData);
		}

		miopenConvBwdDataAlgorithm_t convolution_data_algorithm = (miopenConvBwdDataAlgorithm_t) *dataAlgo;

		//printf("Backward Data Workspace Bytes: %zu\n", workspace_bytes);

		float * temp_in_deriv = NULL;
		
		if (toAdd){
			hipMalloc(&temp_in_deriv, inp_size * sizeof(float));
			hipMemcpy(temp_in_deriv, input_deriv, inp_size * sizeof(float), hipMemcpyDeviceToDevice);
		}

		hipMalloc(&workspace, *memData);

		status = miopenConvolutionBackwardData(trainer -> miopenHandle, &alpha, output_descriptor, out_deriv, kernel_descriptor, weights, convolution_descriptor, convolution_data_algorithm, 
										&beta, input_descriptor, input_deriv, workspace, *memData);

		hipFree(workspace);

		if (toAdd){
			prepareAndDoTensorOp(trainer, "ADD", inp_size, in_spatial_dim, in_filters, batch_size, input_deriv, temp_in_deriv, input_deriv);
			hipFree(temp_in_deriv);
		}
		
		workspace_bytes = 0;
	}

	// static const algo_t algos[] = {
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_0,
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_1,
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_FFT,
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_3,
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
    //      miopen_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    //  };


	// Compute deriv w.r.t filter weights
	//miopenConvAlgoPerf_t top_filter_algo[1];
	//cudnnGetConvolutionBackwardFilterAlgorithm_v7(trainer -> miopenHandle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor, 1, &returned_cnt, top_filter_algo);
	


	if (*filtAlgo == -1){

		miopenConvolutionBackwardWeightsGetWorkSpaceSize(trainer -> miopenHandle, output_descriptor, input_descriptor, convolution_descriptor, kernel_descriptor, &workspace_bytes);
		hipMalloc(&workspace, workspace_bytes);

		min_time = 1000000;

		miopenConvAlgoPerf_t * perfResultsFilt = (miopenConvAlgoPerf_t *) malloc(MAX_REQ_CONV_FIND_ALGO * sizeof(miopenConvAlgoPerf_t));
		miopenFindConvolutionBackwardWeightsAlgorithm(trainer -> miopenHandle, output_descriptor, out_deriv, input_descriptor, input, convolution_descriptor, kernel_descriptor, weight_deriv, MAX_REQ_CONV_FIND_ALGO, &returned_cnt, perfResultsFilt, workspace, workspace_bytes, true);
		
		hipFree(workspace);

		miopenConvAlgoPerf_t cur_perf_results;
		//FILE * fp_perf_filt = fopen("kernel_perfs/backward_filter_conv.text", "a");
		//fprintf(fp_perf_filt, "\nConvolution Arguments: %d,%d,%d,%d,%d,%d\n", in_spatial_dim, kern_dim, stride, in_filters, out_filters, batch_size);
		for (int i = 0; i < returned_cnt; i++){
			cur_perf_results = perfResultsFilt[i];
			//fprintf(fp_perf_filt, "%d,%f,%zu,%d\n", cur_perf_results.algo, cur_perf_results.time, cur_perf_results.memory, cur_perf_results.mathType);
			if (cur_perf_results.time > 0){
				// measured bandthwidth ratio of hipMalloc as MiB / 34 = ms delay
				//total_time = cur_perf_results.time + ((float) cur_perf_results.memory / 1e6) / 34;
				total_time = cur_perf_results.time;
				if (total_time < min_time){
					min_time = total_time;
					min_algo = cur_perf_results.bwd_weights_algo;
					min_mem = cur_perf_results.memory;
				}
			}
		}
		//fclose(fp_perf_filt);
		//free(perfResultsFilt);
		*filtAlgo = min_algo;
		*memFilt = min_mem;
		free(perfResultsFilt);
	}


	hipMalloc(&workspace, *memFilt);

	miopenConvBwdWeightsAlgorithm_t convolution_filter_algorithm = (miopenConvBwdWeightsAlgorithm_t) *filtAlgo;
	
	//printf("Backward Filter Workspace Bytes: %zu\n", workspace_bytes);

	// printf("Filter Workspace Bytes: %zu\n", workspace_bytes);
	// printf("Workspace Bytes Status: %s\n", miopenGetErrorString(status_bytes));

	status = miopenConvolutionBackwardWeights(trainer -> miopenHandle, &alpha, output_descriptor, out_deriv, input_descriptor, input, convolution_descriptor, convolution_filter_algorithm, 
									&beta, kernel_descriptor, weight_deriv, workspace, *memFilt);
	// printf("Back Filt Algo Status: %s\n", miopenGetErrorString(status));

	
	// float * w_deriv_cpu = (float *) malloc(in_filters * out_filters * kern_dim * kern_dim * sizeof(float));
	// hipMemcpy(w_deriv_cpu, weight_deriv, in_filters * out_filters * kern_dim * kern_dim * sizeof(float), hipMemcpyDeviceToHost);

	// int all_zero_w = 1;
	// for (size_t i = 0; i < in_filters * out_filters * kern_dim * kern_dim; i++){
	// 	if (w_deriv_cpu[i] != 0){
	// 		all_zero_w = 0;
	// 		break;
	// 	}
	// }

	// printf("All Zero Weight Deriv?: %d\n", all_zero_w);

	// status = hipGetLastError();
	// printf("Status after backward conv weight deriv: %s\n\n", hipGetErrorString(status));

	hipFree(workspace);
	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyTensorDescriptor(output_descriptor);
	miopenDestroyTensorDescriptor(kernel_descriptor);
	miopenDestroyConvolutionDescriptor(convolution_descriptor);
}

void prepareAndDoBatchNormAndActivate(Train_ResNet * trainer, BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, int batch_size, float eps, float * input, float * output, bool to_activate){
	// reading values from batch norm params
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;

	// read the output device pointers from batch_norm_cache
	float * means_out = batch_norm_cache -> means;
	float * inv_vars_out = batch_norm_cache -> inv_vars;

	float alpha_dummy = 1, beta_dummy = 0;

	miopenTensorDescriptor_t input_descriptor;
	miopenCreateTensorDescriptor(&input_descriptor);
	miopenSet4dTensorDescriptor(input_descriptor, miopenFloat, batch_size, filters, spatial_dim, spatial_dim);

	miopenTensorDescriptor_t bn_descriptor;
	miopenCreateTensorDescriptor(&bn_descriptor);

	miopenBatchNormMode_t bn_mode = miopenBNSpatial;

	miopenDeriveBNTensorDescriptor(bn_descriptor, input_descriptor, bn_mode);

	miopenBatchNormalizationForwardTraining(trainer -> miopenHandle, bn_mode, &alpha_dummy, &beta_dummy, input_descriptor, input, input_descriptor, output, bn_descriptor, gamma, beta, 1, NULL, NULL, trainer -> eps, means_out, inv_vars_out);

	miopenDestroyTensorDescriptor(input_descriptor);
	miopenDestroyTensorDescriptor(bn_descriptor);

	if (to_activate){

		size_t bn_output_size = batch_size * filters * spatial_dim * spatial_dim;

		//dim3 gridDimBN(ceil((float) (bn_output_size) / MAX_THREAD_PER_BLOCK));
		//dim3 blockDimBN(MAX_THREAD_PER_BLOCK);

		//doActivation <<< gridDimBN, blockDimBN >>> (bn_output_size, output, output);
		prepareAndDoActivation(trainer, bn_output_size, output, spatial_dim, filters, batch_size, output);

	}

}

void prepareAndDoActivationAndBatchNormDeriv(Train_ResNet * trainer, BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, BatchNorm * batch_norm_param_derivs,
																								int batch_size, float eps, float * input, float * activated, float * out_layer_deriv, float * input_deriv, bool to_activate_deriv){
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;
	float * means = batch_norm_cache -> means;
	float * inv_vars = batch_norm_cache -> inv_vars;

	float * gamma_deriv = batch_norm_param_derivs -> gamma;
	float * beta_deriv = batch_norm_param_derivs -> beta;

	const float alpha_data = 1, beta_data = 0, alpha_param = 1, beta_param = 0;


	if (to_activate_deriv){
		size_t bn_output_size = batch_size * filters * spatial_dim * spatial_dim;

		// dim3 gridDimBN(ceil((float) (bn_output_size) / MAX_THREAD_PER_BLOCK));
		// dim3 blockDimBN(MAX_THREAD_PER_BLOCK);
		// doActivationDeriv <<< gridDimBN, blockDimBN >>> (bn_output_size, activated, out_layer_deriv, out_layer_deriv);
		prepareAndDoActivationDeriv(trainer, bn_output_size, activated, out_layer_deriv, activated, spatial_dim, filters, batch_size, out_layer_deriv);
	}



	miopenTensorDescriptor_t layer_descriptor;
	miopenCreateTensorDescriptor(&layer_descriptor);
	miopenSet4dTensorDescriptor(layer_descriptor, miopenFloat, batch_size, filters, spatial_dim, spatial_dim);

	miopenTensorDescriptor_t bn_descriptor;
	miopenCreateTensorDescriptor(&bn_descriptor);

	miopenBatchNormMode_t bn_mode = miopenBNSpatial;

	miopenDeriveBNTensorDescriptor(bn_descriptor, layer_descriptor, bn_mode);

	miopenBatchNormalizationBackward(trainer -> miopenHandle, bn_mode, &alpha_data, &beta_data, &alpha_param, &beta_param, 
											layer_descriptor, input, layer_descriptor, out_layer_deriv, layer_descriptor, input_deriv,
											bn_descriptor, gamma, gamma_deriv, beta_deriv, eps, means, inv_vars);

	miopenDestroyTensorDescriptor(layer_descriptor);
	miopenDestroyTensorDescriptor(bn_descriptor);
}


void prepareAndDoMatMulLeftTranspose(const float * left_orig, const float * right, int left_orig_rows, int left_orig_cols, int right_rows, int right_cols, float * out){
	float * temp_left;
	hipMalloc(&temp_left, left_orig_rows * left_orig_cols * sizeof(float));

	dim3 gridDimTranspose(ceil((float) left_orig_rows / TILE_WIDTH), ceil((float)left_orig_cols / TILE_WIDTH));
	dim3 blockDimTranspose(TILE_WIDTH, TILE_WIDTH);
	transpose <<< gridDimTranspose, blockDimTranspose >>> (left_orig, left_orig_rows, left_orig_cols, temp_left);

	dim3 gridDimMatMul(ceil((float) left_orig_cols / TILE_WIDTH), ceil((float) right_cols / TILE_WIDTH));
	dim3 blockDimMatMul(TILE_WIDTH, TILE_WIDTH);
	matMul <<< gridDimMatMul, blockDimMatMul >>> (temp_left, right, left_orig_cols, right_rows, right_cols, out);
	hipFree(temp_left);
}

void prepareAndDoMatMulRightTranspose(const float * left, const float * right_orig, int left_rows, int left_cols, int right_orig_rows, int right_orig_cols, float * out){
	float * temp_right;
	hipMalloc(&temp_right, right_orig_rows * right_orig_cols * sizeof(float));
	
	dim3 gridDimTranspose(ceil((float) right_orig_rows / TILE_WIDTH), ceil((float)right_orig_cols / TILE_WIDTH));
	dim3 blockDimTranspose(TILE_WIDTH, TILE_WIDTH);

	transpose <<< gridDimTranspose, blockDimTranspose >>> (right_orig, right_orig_rows, right_orig_cols, temp_right);
	
	dim3 gridDimMatMul(ceil((float) left_rows / TILE_WIDTH), ceil((float) right_orig_rows / TILE_WIDTH));
	dim3 blockDimMatMul(TILE_WIDTH, TILE_WIDTH);
	matMul <<< gridDimMatMul, blockDimMatMul >>> (left, temp_right, left_rows, left_cols, right_orig_rows, out);
	hipFree(temp_right);
}

void printDeviceData(const char * name_of_variable, float * device_variable, int size){
	bool print = TO_PRINT;
	if (print){
		float * cpu_data = (float *) malloc(size * sizeof(float));
		hipMemcpy(cpu_data, device_variable, size * sizeof(float), hipMemcpyDeviceToHost);
		printf("VARIABLE NAME: %s\n\n", name_of_variable);
		printf("DATA:\n");
		for (int i = 0; i < size; i++){
			printf("%d: %f\n", i, cpu_data[i]);
		}
		printf("\n\n\n");
		free(cpu_data);
	}
}

void forward_pass(Train_ResNet * trainer){

	Dims * dims = trainer -> model -> dims;

	float eps = trainer -> eps;
	int batch_size = trainer -> batch_size;

	int conv_algo_ind = 0;
	int * conv_algos = trainer -> forward_buffer -> conv_algos;
	size_t * workspace_algos = trainer -> forward_buffer -> workspace_algos;

	float * input = trainer -> cur_batch -> images;
	float * first_conv = trainer -> model -> params -> init_conv_layer;
	float * first_conv_output = trainer -> forward_buffer -> activations -> init_conv_applied;
	// first apply the convolutions
	// launch grid dimensions as (OUT_SPATIAL_DIM, OUT_SPATIAL_DIM, OUT_FILTER_CHUNK) blocks, and launch with block dim as (out_filt_rows_shared, sub_batch) threads
	
	// 3 colors
	int init_in_filters = 3;
	int init_spatial_dim = dims -> input;
	int init_kernel_dim = dims -> init_kernel_dim;
	int init_out_filters = dims -> init_conv_filters;
	int init_stride = dims -> init_conv_stride;
	int init_out_spatial_dim = init_spatial_dim / init_stride;

	prepareAndDoConvolution(trainer, init_spatial_dim, init_kernel_dim, init_in_filters, init_out_filters, init_stride, batch_size, input, first_conv, first_conv_output, &conv_algos[conv_algo_ind], &workspace_algos[conv_algo_ind]);
	conv_algo_ind++;

	int print_size = 10;
	printDeviceData("INIT CONV APPLIED", first_conv_output, print_size);

	BatchNorm * norm_init_conv_params = trainer -> model -> params -> norm_init_conv;
	Cache_BatchNorm * norm_init_conv_cache = trainer -> forward_buffer -> activations -> norm_init_conv;
	float * init_activated = trainer -> forward_buffer -> activations -> init_conv_activated;

	prepareAndDoBatchNormAndActivate(trainer, norm_init_conv_params, norm_init_conv_cache, batch_size, eps, first_conv_output, init_activated, true);

	printDeviceData("INIT CONV ACTIVATED", init_activated, print_size);

	int init_maxpool_dim = dims -> init_maxpool_dim;
	int init_maxpool_stride = dims -> init_maxpool_stride;
	float * init_convblock_input = trainer -> forward_buffer -> activations -> init_convblock_input;

	prepareAndDoPool(trainer, "MAX", init_activated, init_spatial_dim / init_stride, init_out_filters, init_maxpool_dim, init_maxpool_stride, batch_size, init_convblock_input);

	printDeviceData("MAX POOL OUTPUT", init_convblock_input, print_size);

	/* NOW CAN MOVE ONTO TO CONV_BLOCK LAYERS! */

	int n_conv_blocks = dims -> n_conv_blocks;

	
	ConvBlock ** params_conv_blocks = trainer -> model -> params -> conv_blocks;
	Activation_ConvBlock ** activation_conv_blocks = trainer -> forward_buffer -> activations -> activation_conv_blocks;
	ConvBlock * cur_conv_block_params;
	Activation_ConvBlock * cur_conv_block_activation;
	int in_spatial_dim, kern_dim, in_filters, out_filters, stride, out_spatial_dim, total_size_conv_block_output;

	float * conv_block_input = init_convblock_input;
	float *conv_input, * conv_weights, * conv_output, *norm_input, * norm_output, * conv_block_output, * conv_block_output_activated;
	float *projection_weights, *transformed_residual, *post_projection_norm_vals;
	BatchNorm * cur_batch_norm_params;
	Cache_BatchNorm * cur_batch_norm_cache;
	for (int i = 0; i < n_conv_blocks; i++){

		cur_conv_block_params = params_conv_blocks[i];
		cur_conv_block_activation = activation_conv_blocks[i];

		// do first 1x1 depth_reduce convolution
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> reduced_depth;
		kern_dim = 1;
		stride = 1;
		// either intialized first time above loop from the maxpool
		// every other block will be the normalized, activated output of previous conv block (previous iteration output) 
		conv_input = conv_block_input;
		conv_weights = cur_conv_block_params -> depth_reduction;
		conv_output = cur_conv_block_activation -> post_reduced;


		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output, &conv_algos[conv_algo_ind], &workspace_algos[conv_algo_ind]);
		conv_algo_ind++;

		printDeviceData("REDUCED CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_depth_reduction;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_reduced;
		norm_output = cur_conv_block_activation -> post_reduced_activated;

		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, true);

		printDeviceData("REDUCED CONV NORM & ACTIVATED", norm_output, print_size);

		// do 3x3 spatial convolution

		// same as in first conv
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		// now is output filters of 1st conv, which is reduced depth filters
		in_filters = cur_conv_block_params -> reduced_depth;
		// keeps depth the same, just spatial conv
		out_filters = cur_conv_block_params -> reduced_depth;
		kern_dim = 3;
		// if stride is occurring in conv block happens at this kernel
		stride = cur_conv_block_params -> stride;
		conv_input = norm_output;
		conv_weights = cur_conv_block_params -> spatial;;
		conv_output = cur_conv_block_activation -> post_spatial;

		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output, &conv_algos[conv_algo_ind], &workspace_algos[conv_algo_ind]);
		conv_algo_ind++;

		printDeviceData("SPATIAL CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_spatial;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_spatial;
		norm_output = cur_conv_block_activation -> post_spatial_activated;

		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, true);

		printDeviceData("SPATIAL CONV NORM & ACTIVATED", norm_output, print_size);

		// do 1x1 depth expansion convolution

		// if stride happened now would need to take that into account
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		// prev 3x3 conv kept out filters as reduced depth
		in_filters = cur_conv_block_params -> reduced_depth;
		// now creating expanded depth out filters
		out_filters = cur_conv_block_params -> expanded_depth;
		kern_dim = 1;
		stride = 1;
		conv_input = norm_output;
		conv_weights = cur_conv_block_params -> depth_expansion;
		conv_output = cur_conv_block_activation -> post_expanded;

		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output, &conv_algos[conv_algo_ind], &workspace_algos[conv_algo_ind]);
		conv_algo_ind++;

		printDeviceData("EXPANDED CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_expansion;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_expanded;
		norm_output = cur_conv_block_activation -> output;

		// do not activate because first need to add to (projection) residual
		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, false);

		printDeviceData("EXPANDED NORM & ACTIVATED", norm_output, print_size);

		// now need to add identity of conv_block_input (if same dimensions), or project=convolve (different dimensions) and add to conv_output
		// projection is a incoming block filters X expanded depth matrix
		// if stride of 2 in additon to depth change, then 3x3 kernel with stride 2 applied to block input
		// works as a depth-wise 1x1 convolution where in_filters = incoming_filters and out_filters = expanded_depth

		// already updated
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim);
		out_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> expanded_depth;
		stride = cur_conv_block_params -> stride;
		if (stride == 2){
			kern_dim = 3;
		}
		else{
			kern_dim = 1;
		}
		projection_weights = cur_conv_block_params -> projection;

		total_size_conv_block_output = out_spatial_dim * out_spatial_dim * out_filters * batch_size;
		
				
		// the conv_block initializer already handled if we need projection, and if so it allocated weights
		// if there is a projection needed we will do convolution with the above parameters
		if (projection_weights){
			// allocated device memory to store output
			transformed_residual = cur_conv_block_activation -> transformed_residual;
			prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_block_input, projection_weights, transformed_residual, &conv_algos[conv_algo_ind], &workspace_algos[conv_algo_ind]);
			conv_algo_ind++;
			post_projection_norm_vals = cur_conv_block_activation -> post_projection_norm_vals;
			prepareAndDoBatchNormAndActivate(trainer, cur_conv_block_params -> norm_projection, cur_conv_block_activation -> norm_post_projection, batch_size, eps, transformed_residual, post_projection_norm_vals, false);
		}
		else{
			// would've been null, so renaming for semantic clarity
			post_projection_norm_vals = conv_block_input;
		}

		printDeviceData("(TRANSFORMED) RESIDUAL", transformed_residual, print_size);

		dim3 gridDimConvOutput(ceil((float) total_size_conv_block_output / MAX_THREAD_PER_BLOCK));
		dim3 blockDimConvOutput(MAX_THREAD_PER_BLOCK);

		conv_block_output = cur_conv_block_activation -> output;
		// add identity residual connection (or projected residual connection) to the prior batch norm output
		//addVec <<< gridDimConvOutput, blockDimConvOutput >>> (total_size_conv_block_output, norm_output, post_projection_norm_vals, conv_block_output);
		//hipblasSaxpy(total_size_conv_block_output, 1.0, post_projection_norm_vals, 1, conv_block_output, 1);
		prepareAndDoTensorOp(trainer, "ADD", total_size_conv_block_output, out_spatial_dim, out_filters, batch_size, conv_block_output, post_projection_norm_vals, conv_block_output);

		// activated output from previous block not needed anymore
		if (i != 0) {
			hipFree(conv_block_input);
		}

		printDeviceData("CONV OUTPUT + (TRANSFORMED) RESIDUAL", conv_block_output, print_size);

		hipMalloc(&conv_block_output_activated, total_size_conv_block_output * sizeof(float));

		//doActivation <<< gridDimConvOutput, blockDimConvOutput >>> (total_size_conv_block_output, conv_block_output, conv_block_output_activated);

		prepareAndDoActivation(trainer, total_size_conv_block_output, conv_block_output, out_spatial_dim, out_filters, batch_size, conv_block_output_activated);

		printDeviceData("CONV OUTPUT ACTIVATED", conv_block_output, print_size);
		
		// prepare for next block...
		conv_block_input = conv_block_output_activated;
	}

	int final_filters = dims -> final_depth;
	int final_spatial_dim = params_conv_blocks[n_conv_blocks - 1] -> incoming_spatial_dim;
	float * final_conv_block_output = conv_block_output_activated;
	float * final_avg_pool_values = trainer -> forward_buffer -> activations -> final_conv_output_pooled;

	// NEED TO DO AVERAGE POOL OF LAST LAYER to go from (batch_size, 7, 7, 2048) to (batch size, 1, 1, 2048)

	// format of output is each row is a sample and has a row size of 2048
	dim3 gridDimAvgPool(final_filters);
	dim3 blockDimAvgPool(batch_size);
	doFilterAvgPool <<< gridDimAvgPool, blockDimAvgPool >>> (final_conv_block_output, final_spatial_dim, final_avg_pool_values);
	// prepareAndDoPool(trainer, "AVG", final_conv_block_output, final_spatial_dim, final_filters, 7, 1, batch_size, final_avg_pool_values);

	// clean up the activated version of last conv block
	hipFree(conv_block_output_activated);

	printDeviceData("FINAL AVG POOL VALUES", final_avg_pool_values, print_size);


	// APPLY FULLY CONNECTED LAYER BETWEEN (2048, 1000)
	float * fc_weights = trainer -> model -> params -> fully_connected;
	float * fc_output = trainer -> forward_buffer -> activations -> linear_output;
	int output_dim = dims -> output;

	// matrix multiply between (N, 2048) and fc weights of (2048, 1000), yields output of (N, 1000)
	// output is each row is a unique sample

	// GRID has dim (OUT_ROWS / TILE_WIDTH, OUT_COLS/TILE_WIDTH)
	// each BLOCK has dim (TILE_WIDTH, TILE_WIDTH)
	dim3 gridDimFCOutput(ceil((float) batch_size / TILE_WIDTH), ceil((float) output_dim / TILE_WIDTH));
	dim3 blockDimFCOutput(TILE_WIDTH, TILE_WIDTH);

	matMul <<< (gridDimFCOutput), (blockDimFCOutput) >>> (final_avg_pool_values, fc_weights, batch_size, final_filters, output_dim, fc_output);

	printDeviceData("FULLY CONNECTED WEIGHTS", fc_weights, print_size);
	printDeviceData("FULLY CONNECTED OUTPUT", fc_output, print_size);

	// DO SOFTMAX
	float * pred = trainer -> forward_buffer -> pred;
	dim3 gridDimSoftMax(1);
	dim3 blockDimSoftMax(batch_size);
	softMax <<< gridDimSoftMax, blockDimSoftMax >>> (fc_output, batch_size, output_dim, pred);

	printDeviceData("SOFTMAX PREDICTIONS", pred, print_size);

	// FINISH UP BY POPULATING PREDICTIONS ONTO CPU
	float * pred_cpu = trainer -> forward_buffer -> pred_cpu;
	hipMemcpy(pred_cpu, pred, batch_size * output_dim * sizeof(float), hipMemcpyDeviceToHost);
}

void backwards_pass(Train_ResNet * trainer){
	
	Dims * dims = trainer -> model -> dims;
	int batch_size = trainer -> batch_size;
	int output_dim = dims -> output;
	float eps = trainer -> eps;
	Activations * activations = trainer -> forward_buffer -> activations;
	Params * model_params = trainer -> model -> params;
	Backprop_Buffer * backprop_buffer = trainer -> backprop_buffer;
	Params * param_derivs = backprop_buffer -> param_derivs;

	size_t max_activation_size = activations -> max_activation_size;

	float * activ_deriv_buff;
	hipMalloc(&activ_deriv_buff, max_activation_size * sizeof(float));

	float * block_activ_deriv;
	hipMalloc(&block_activ_deriv, max_activation_size * sizeof(float));

	float * temp_deriv_buff;
	hipMalloc(&temp_deriv_buff, max_activation_size * sizeof(float));

	float * prev_conv_block_out_deriv;
	hipMalloc(&prev_conv_block_out_deriv, max_activation_size * sizeof(float));

	// hipMemset(activ_deriv_buff, 0, max_activation_size * sizeof(float));
	// hipMemset(prev_conv_block_out_deriv, 0, max_activation_size * sizeof(float));
	// hipMemset(block_activ_deriv, 0, max_activation_size * sizeof(float));
	// hipMemset(temp_deriv_buff, 0, max_activation_size * sizeof(float));

	int conv_algo_ind = 0;
	int *data_algos = backprop_buffer -> data_algos;
	int *filt_algos = backprop_buffer -> filt_algos;
	size_t *workspace_data = backprop_buffer -> workspace_data;
	size_t *workspace_filt = backprop_buffer -> workspace_filt;

	/* STEP 1: LAST LAYER DERIVATIVE */

	// layer has output_dim * batch_size values
	// End of network was: fully connected layer -> softmax
	// Derivative of cross entropy loss w.r.t to fully connected values is: s - y where s is softmax value
	// thus copy softmax values and subtract 1 from the correct index (we know labels y are 0 except correct label of 1)
	int * correct_classes = trainer -> cur_batch -> correct_classes;
	float * pred = trainer -> forward_buffer -> pred;
	float * output_layer_deriv = backprop_buffer -> output_layer_deriv;
	hipMemcpy(output_layer_deriv, pred, batch_size * output_dim * sizeof(float), hipMemcpyDeviceToDevice);

	dim3 gridDimCrossDeriv(1);
	dim3 blockDimCrossDeriv(batch_size);
	crossEntropyDeriv <<< gridDimCrossDeriv, blockDimCrossDeriv >>> (output_layer_deriv, correct_classes, output_dim, batch_size);

	// divide by the batch size because loss is sum across all batches...
	// NOT SURE IF WE WANT TO DO AVERAGE HERE OR NOT...?
	
	// dim3 gridDimTakeAvgDeriv(output_dim);
	// dim3 blockDimTakeAvgDeriv(batch_size);
	// averageDerivOverBatchSize <<< gridDimTakeAvgDeriv, blockDimTakeAvgDeriv >>> (output_layer_deriv, output_dim, batch_size);

	/* STEP 2: FC WEIGHT DERIV AND FINAL AVG POOL (SECOND LAST ACTIVTION LAYER) DERIVATIVE */

	// TODO: MAKE SURE THE DIMENSIONS ARE CORRECT ORDER...

	// FC WEIGHTS (2048, 1000) DERIV = matMul(transpose(final_conv_output_pooled), output_layer_deriv)
	int final_depth = dims -> final_depth;
	float * fc_deriv = param_derivs -> fully_connected;
	float * final_conv_output_pooled = activations -> final_conv_output_pooled;
	prepareAndDoMatMulLeftTranspose(final_conv_output_pooled, output_layer_deriv, batch_size, final_depth, batch_size, output_dim, fc_deriv);

	int print_size = 10;
	printDeviceData("FC WEIGHT DERIV", fc_deriv, print_size);

	// FINAL AVG POOL (N, 2048) DERIV = matMul(output_layer_deriv, transpose(FC Weight))
	float * fc_weights = model_params -> fully_connected;
	float * final_avg_pool_deriv = activ_deriv_buff;
	prepareAndDoMatMulRightTranspose(output_layer_deriv, fc_weights, batch_size, output_dim, final_depth, output_dim, final_avg_pool_deriv);

	printDeviceData("FINAL AVG POOL ACTIVATION DERIV", final_avg_pool_deriv, print_size);


	/* CONV BLOCK DATA FROM FORWARD PASS */
	int n_conv_blocks = dims -> n_conv_blocks;
	Activation_ConvBlock ** activation_conv_blocks = activations -> activation_conv_blocks;
	ConvBlock ** conv_block_params = model_params -> conv_blocks;

	/* CONV BLOCK DERIV BUFFERS */
	ConvBlock ** conv_block_param_derivs = param_derivs -> conv_blocks;


	int final_spatial_dim = conv_block_params[n_conv_blocks - 1] -> incoming_spatial_dim;
	
	/* STEP 3: AVG POOL DERIV */

	// get the location for the deriv of final conv block output
	float * final_conv_block_output_deriv = temp_deriv_buff;
	// using final_avg_pool_deriv (batch_size, 2048) to populate final_conv_block_output_deriv (batch_size, 7, 7, 2048)
	// each expanded (prior to pooling) spatial index takes on value of given filter's avg_pool_deriv / (spatial_dim^2)
	dim3 gridDimAvgPoolDeriv(final_depth);
	dim3 blockDimAvgPoolDeriv(batch_size);
	filterAvgPoolDeriv <<< gridDimAvgPoolDeriv, blockDimAvgPoolDeriv >>> (final_avg_pool_deriv, final_depth, batch_size, final_spatial_dim, final_conv_block_output_deriv);

	//prepareAndDoPoolDeriv(trainer, "AVG", activation_conv_blocks[n_conv_blocks - 1] -> output, activations -> final_conv_output_pooled, activ_deriv_buff, final_spatial_dim, final_depth, 7, 1, batch_size, final_conv_block_output_deriv);

	hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

	printDeviceData("FINAL CONV BLOCK OUTPUT ACTIVATION DERIV", final_conv_block_output_deriv, print_size);

	
	/* STEP 4: CONV BLOCK & BATCH NORM DERIVS  */
	

	// we are starting with deriv of last conv block output...

	// To go backwards for each block we:
		// 1.) Get deriv of output activated (ReLU so just 0 or 1)
		// 2.) Get deriv projection filter & transformed (if there is a projection of residual, otherwise both derivs are 1)
		// 3.) Multiply the deriv of output activation * deriv of transformed residual and add to the deriv of first layer of conv block (= output activated of prior block)
		// 4.) Multiply the deriv of output activation * deriv of batch norm for expanded conv output (with respect to both its own parameters and also the input to batch norm = expanded conv output)
		// 5.) Get deriv of expanded convolution & deriv of input to expanded convolution (= batch norm output of spatial conv)
		// 6.) Get deriv of batch norm for spatial conv output (with respect to both its own parameters and also the input to batch norm = spatial conv output)
		// 7.) Get deriv of sptial convolution & deriv of input to spatial convolution (= batch norm output of reduced conv)
		// 8.) Get deriv of batch norm for reduced conv output (with respect to both its own parameters and also the input to batch norm = reduced conv output)
		// 9.) Get deriv of reduced convolution & deriv of input to reduced convolution, which is the first layer of conv block (= batch norm output of prior conv block)
		// Items 3.) and 9.) provide the derivative used to repeat process for prior block

	

	// will update these variables throughout loop to pass to batch norm deriv
	float *bn_input, *bn_activated, *bn_out_layer_deriv, *bn_input_deriv;
	BatchNorm *cur_batch_norm_params, *cur_batch_norm_param_derivs;
	Cache_BatchNorm *cur_batch_norm_cache;

	// will update these every iteration through conv_blocks
	ConvBlock * cur_conv_block_params, *cur_conv_block_param_derivs;
	Activation_ConvBlock * cur_conv_block_activation;

	// will update these within every iteration through conv_blocks
	// because multiple convolutions per block, but keep params same for easy calls to functions
	int in_spatial_dim, kern_dim, in_filters, out_filters, stride;
	float *conv_input, *conv_weight, *conv_out_deriv;
	float *conv_input_deriv, *conv_weight_deriv;


	// STARTING POINT FROM BACKPROP COMING FROM UPSTREAM LAYERS IS AT LAST CONV BLOCK ACTIVATION -> OUTPUT_ACTIVATED
	float *conv_block_input, *conv_block_input_deriv, * upstream_deriv, *block_activation_deriv, *final_output_pre_activ;
	float *temp_conv_inp_activated;
	float *temp_ptr;
	size_t conv_input_size;

	// extra temp variables
	int total_size, output_size;

	for (int i = n_conv_blocks - 1; i >= 0; i--){

		// residual deriv and normal backprop deriv added to this
		if (i == 0){
			conv_block_input = activations -> init_convblock_input;
			conv_block_input_deriv = prev_conv_block_out_deriv;
		}
		else{
			conv_block_input = activation_conv_blocks[i - 1] -> output;
			conv_block_input_deriv = prev_conv_block_out_deriv;
		}

		// getting current conv block parameters and buffers to hold derivs
		cur_conv_block_params = conv_block_params[i];
		cur_conv_block_param_derivs = conv_block_param_derivs[i];

		// getting current conv block activation values and buffers to hold derivs
		cur_conv_block_activation = activation_conv_blocks[i];

		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim);
		in_filters = cur_conv_block_params -> incoming_filters;
		conv_input_size = in_spatial_dim * in_spatial_dim * in_filters * batch_size;
		hipMalloc(&temp_conv_inp_activated, conv_input_size * sizeof(float));
		//hipMemset(temp_conv_inp_activated, 0, conv_input_size * sizeof(float));
		// repeat the activation because not stored

		// dim3 gridDimReActiv(ceil((float) conv_input_size / MAX_THREAD_PER_BLOCK));
		// dim3 blockDimReActiv(MAX_THREAD_PER_BLOCK);
		// doActivation <<< gridDimReActiv, blockDimReActiv >>> (conv_input_size, conv_block_input, temp_conv_inp_activated);
		prepareAndDoActivation(trainer, conv_input_size, conv_block_input, in_spatial_dim, in_filters, batch_size, temp_conv_inp_activated);



		/* 1: Conv Block Output Activation */
		
		// GIVEN
		upstream_deriv = activ_deriv_buff;
		final_output_pre_activ = cur_conv_block_activation -> output;

		// to fill in the ReLU deriv location
		block_activation_deriv = block_activ_deriv;

		output_size = batch_size * cur_conv_block_params -> expanded_depth * cur_conv_block_params -> incoming_spatial_dim * cur_conv_block_params -> incoming_spatial_dim / ((cur_conv_block_params -> stride) * (cur_conv_block_params -> stride));

		// dim3 gridDimOutput(ceil((float) output_size / MAX_THREAD_PER_BLOCK));
		// dim3 blockDimOutput(MAX_THREAD_PER_BLOCK);
		// doActivationDeriv <<< gridDimOutput, blockDimOutput >>> (output_size, final_output_pre_activ, upstream_deriv, block_activation_deriv);

		// float * temp_conv_out_activated;
		// hipMalloc(&temp_conv_out_activated, output_size * sizeof(float));

		// prepareAndDoActivation(trainer, output_size, final_output_pre_activ, cur_conv_block_params -> incoming_spatial_dim / (cur_conv_block_params -> stride), cur_conv_block_params -> expanded_depth, batch_size, temp_conv_out_activated);

		prepareAndDoActivationDeriv(trainer, output_size, final_output_pre_activ, upstream_deriv, final_output_pre_activ, cur_conv_block_params -> incoming_spatial_dim / (cur_conv_block_params -> stride), cur_conv_block_params -> expanded_depth, batch_size, block_activation_deriv);

		//hipFree(temp_conv_out_activated);

		hipMemcpy(activ_deriv_buff, block_activ_deriv, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);


		/* 2: (Transformed) Residual Derivs & Chained/Added to Conv Block Input Deriv (= prior_block_output_deriv) */

		// check if there is a projection (aka convolution over depth/kern_dim=1 or possibly stride=2/kern_dim=3), otherwise the projection deriv is 1
		// If there is a projection need to compute derivative of the projection convolution kernel weights and deriv w.r.t. projection convolution input=conv_block_input=prior_block_output_activated
		if (cur_conv_block_params -> projection){


			// DEAL WITH BATCH NORM
			// update the current batch norm layer pointers
			cur_batch_norm_params = cur_conv_block_params -> norm_projection;
			cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_projection;
			cur_batch_norm_cache = cur_conv_block_activation -> norm_post_projection;

			// fill in details about backprop I/O
			// dL/dBN_Output (given)
			bn_out_layer_deriv = activ_deriv_buff;
			// dL/dBN_Input (to fill in)
			bn_input_deriv = temp_deriv_buff;
			// input to batch norm layer from forward pass
			bn_input = cur_conv_block_activation -> transformed_residual;
			// activated output of batch norm layer from forward pass
			bn_activated = cur_conv_block_activation -> post_projection_norm_vals;
		
			prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, false);

			temp_ptr = temp_deriv_buff;
			temp_deriv_buff = activ_deriv_buff;
			activ_deriv_buff = temp_ptr;

			//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);


			// CONVOLUTION DIMENSIONS
			in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim);
			in_filters = cur_conv_block_params -> incoming_filters;
			out_filters = cur_conv_block_params -> expanded_depth;
			stride = cur_conv_block_params -> stride;
			if (stride == 2){
				kern_dim = 3;
			}
			else{
				kern_dim = 1;
			}


			// CONVOLUTION FORWARD DATA
			// transformed residual convolution input is the value at first step of conv block => activated output from previous block
			
			conv_input = temp_conv_inp_activated;
			conv_weight = cur_conv_block_params -> projection;
			// from backprop
			conv_out_deriv = activ_deriv_buff;

			// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
			// because residual
			conv_input_deriv = conv_block_input_deriv;
			conv_weight_deriv = cur_conv_block_param_derivs -> projection;

			prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true, &data_algos[conv_algo_ind], &filt_algos[conv_algo_ind], &workspace_data[conv_algo_ind], &workspace_filt[conv_algo_ind]);
			conv_algo_ind++;

			printDeviceData("PROJECTED CONV INPUT DERIV", conv_input_deriv, print_size);
			printDeviceData("PROJECTED CONV WEIGHT DERIV", conv_weight_deriv, print_size);
		}
		else{
			total_size = batch_size * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_filters);

			// dim3 gridDimResidual(ceil((float) total_size / MAX_THREAD_PER_BLOCK));
			// dim3 blockDimResidual(MAX_THREAD_PER_BLOCK);
			// setVal <<< gridDimResidual, blockDimResidual >>> (total_size, 0, conv_block_input_deriv);
			//addVec <<< gridDimResidual, blockDimResidual >>> (total_size, conv_block_input_deriv, block_activ_deriv, conv_block_input_deriv);

			//hipblasSaxpy(total_size, 1.0, block_activ_deriv, 1, conv_block_input_deriv, 1);
			hipMemcpy(conv_block_input_deriv, block_activ_deriv, total_size * sizeof(float), hipMemcpyDeviceToDevice);
			
			
		}
		

		/* 3: Expanded Convolution And Batch Norm Derivs */
		int exp_spatial_dim = cur_conv_block_params -> incoming_spatial_dim / cur_conv_block_params -> stride;

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_expansion;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_expansion;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_expanded;

		size_t cur_bn_inp_size = cur_batch_norm_cache -> input_size;
		size_t cur_bn_feature_size = cur_batch_norm_cache -> feature_size;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = block_activ_deriv;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = temp_deriv_buff;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_expanded;
		// activated output of batch norm layer from forward pass

		float * temp_bn_activ;
		hipMalloc(&temp_bn_activ, cur_bn_inp_size * sizeof(float));

		if (cur_conv_block_params -> projection){
			//subVec <<< ceil((float) cur_bn_inp_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (cur_bn_inp_size, cur_conv_block_activation -> output, cur_conv_block_activation -> post_projection_norm_vals, temp_bn_activ);
			//hipblasSaxpy(cur_bn_inp_size, -1.0, cur_conv_block_activation -> post_projection_norm_vals, 1, temp_bn_activ, 1);
			prepareAndDoTensorOp(trainer, "SUB", cur_bn_inp_size, exp_spatial_dim, cur_bn_feature_size, batch_size, cur_conv_block_activation -> output, cur_conv_block_activation -> post_projection_norm_vals, temp_bn_activ);
		}
		else{
			//subVec <<< ceil((float) cur_bn_inp_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (cur_bn_inp_size, cur_conv_block_activation -> output, temp_conv_inp_activated, temp_bn_activ);
			//hipblasSaxpy(cur_bn_inp_size, -1.0, temp_conv_inp_activated, 1, temp_bn_activ, 1);
			prepareAndDoTensorOp(trainer, "SUB", cur_bn_inp_size, exp_spatial_dim, cur_bn_feature_size, batch_size, cur_conv_block_activation -> output, temp_conv_inp_activated, temp_bn_activ);
		}

		bn_activated = temp_bn_activ;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, false);

		hipFree(temp_bn_activ);

		temp_ptr = temp_deriv_buff;
		temp_deriv_buff = activ_deriv_buff;
		activ_deriv_buff = temp_ptr;

		//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);



		printDeviceData("CONV BLOCK OUTPUT ACTIVATION & NORM DERIV", bn_input_deriv, print_size);

		// CONVOLUTION DIMENSIONS
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		in_filters = cur_conv_block_params -> reduced_depth;
		out_filters = cur_conv_block_params -> expanded_depth;
		stride = 1;
		kern_dim = 1;

		// CONVOLUTION FORWARD DATA
		conv_input = cur_conv_block_activation -> post_spatial_activated;
		conv_weight = cur_conv_block_params -> depth_expansion;
		// from backprop
		conv_out_deriv = activ_deriv_buff;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = temp_deriv_buff;
		conv_weight_deriv = cur_conv_block_param_derivs -> depth_expansion;

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true, &data_algos[conv_algo_ind], &filt_algos[conv_algo_ind], &workspace_data[conv_algo_ind], &workspace_filt[conv_algo_ind]);
		conv_algo_ind++;

		temp_ptr = temp_deriv_buff;
		temp_deriv_buff = activ_deriv_buff;
		activ_deriv_buff = temp_ptr;

		//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);
		
		printDeviceData("EXPANDED CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("EXPANDED CONV WEIGHT DERIV", conv_weight_deriv, print_size);


		/* 4: Spatial Convolution Activation and Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_spatial;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_spatial;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_spatial;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = activ_deriv_buff;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = temp_deriv_buff;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_spatial;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> post_spatial_activated;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);

		temp_ptr = temp_deriv_buff;
		temp_deriv_buff = activ_deriv_buff;
		activ_deriv_buff = temp_ptr;
		//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

		printDeviceData("SPATIAL ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

		/* 5: Spatial Convolution Derivs */

		// CONVOLUTION DIMENSIONS
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> reduced_depth;
		out_filters = cur_conv_block_params -> reduced_depth;
		stride = cur_conv_block_params -> stride;
		kern_dim = 3;

		// CONVOLUTION FORWARD DATA
		conv_input = cur_conv_block_activation -> post_reduced_activated;
		conv_weight = cur_conv_block_params -> spatial;
		// from backprop
		conv_out_deriv = activ_deriv_buff;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = temp_deriv_buff;
		conv_weight_deriv = cur_conv_block_param_derivs -> spatial;

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true, &data_algos[conv_algo_ind], &filt_algos[conv_algo_ind], &workspace_data[conv_algo_ind], &workspace_filt[conv_algo_ind]);
		conv_algo_ind++;

		temp_ptr = temp_deriv_buff;
		temp_deriv_buff = activ_deriv_buff;
		activ_deriv_buff = temp_ptr;

		//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

		printDeviceData("SPATIAL CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("SPATIAL CONV WEIGHT DERIV", conv_weight_deriv, print_size);

		/* 6: Reduced Convolution Activation and Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_depth_reduction;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_depth_reduction;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_reduced;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = activ_deriv_buff;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = temp_deriv_buff;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_reduced;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> post_reduced_activated;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);

		temp_ptr = temp_deriv_buff;
		temp_deriv_buff = activ_deriv_buff;
		activ_deriv_buff = temp_ptr;

		//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

		printDeviceData("REDUCED ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

		/* 7: Reduced Convolution Derivs */


		// CONVOLUTION DIMENSIONS
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> reduced_depth;
		stride = 1;
		kern_dim = 1;

		// CONVOLUTION FORWARD DATA
		conv_input = temp_conv_inp_activated;
		conv_weight = cur_conv_block_params -> depth_reduction;
		// from backprop
		conv_out_deriv = activ_deriv_buff;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = conv_block_input_deriv;
		conv_weight_deriv = cur_conv_block_param_derivs -> depth_reduction;

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, true,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv,  true, &data_algos[conv_algo_ind], &filt_algos[conv_algo_ind], &workspace_data[conv_algo_ind], &workspace_filt[conv_algo_ind]);
		conv_algo_ind++;


		printDeviceData("REDUCED CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("REDUCED CONV WEIGHT DERIV", conv_weight_deriv, print_size);

		hipMemcpy(activ_deriv_buff, prev_conv_block_out_deriv, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

		hipMemset(prev_conv_block_out_deriv, 0, max_activation_size * sizeof(float));
		//hipMemset(block_activ_deriv, 0, max_activation_size * sizeof(float));
		//hipMemset(temp_deriv_buff, 0, max_activation_size * sizeof(float));
		hipFree(temp_conv_inp_activated);
	}


	/* STEP 5: MAX POOL DERIV */

	// maxpool dimensions (used in forward pass)
	int maxpool_kern_dim = dims -> init_maxpool_dim;
	int maxpool_stride = dims -> init_maxpool_stride;
	int maxpool_in_spatial_dim = dims -> input / dims -> init_conv_stride;
	int maxpool_out_spatial_dim = maxpool_in_spatial_dim / maxpool_stride;
	int maxpool_filters = dims -> init_conv_filters;

	// backprop up through the init convblock input has been done. the gradient is at:
	float * maxpool_out_deriv = activ_deriv_buff;

	// populating the gradient of input to max_pool located at:
	float * maxpool_inp_deriv = temp_deriv_buff;
	// ensure that gradient has 0's, so that maxPoolDeriv kernel can overwrite only at max ind locations
	int maxpool_inp_size = maxpool_in_spatial_dim * maxpool_in_spatial_dim * maxpool_filters * batch_size;
	hipMemset(maxpool_inp_deriv, 0, maxpool_inp_size * sizeof(float));

	float * max_pool_inp = trainer -> forward_buffer -> activations -> init_conv_activated;
	float * max_pool_out = trainer -> forward_buffer -> activations -> init_convblock_input;

	prepareAndDoPoolDeriv(trainer, "MAX", max_pool_inp, max_pool_out, maxpool_out_deriv, maxpool_in_spatial_dim, maxpool_filters, maxpool_kern_dim, maxpool_stride, batch_size, maxpool_inp_deriv);

	temp_ptr = temp_deriv_buff;
	temp_deriv_buff = activ_deriv_buff;
	activ_deriv_buff = temp_ptr;

	//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

	printDeviceData("MAX POOL INPUT ACTIVATION DERIV", maxpool_inp_deriv, print_size);

	/* STEP 6: INIT BATCH NORM & CONV DERIV */

	// BACKPROP OVER THE BATCH NORM OF FIRST CONV LAYER

	// update the current batch norm layer pointers
	cur_batch_norm_params = model_params -> norm_init_conv;
	cur_batch_norm_param_derivs = param_derivs -> norm_init_conv;
	cur_batch_norm_cache = activations -> norm_init_conv;

	// fill in details about backprop I/O
	// dL/dBN_Output (given)
	bn_out_layer_deriv = activ_deriv_buff;
	// dL/dBN_Input (to fill in)
	bn_input_deriv = temp_deriv_buff;
	// input to batch norm layer from forward pass
	bn_input = activations -> init_conv_applied;
	// activated output of batch norm layer from forward pass
	bn_activated = activations -> init_conv_activated;
		
	prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);
	temp_ptr = temp_deriv_buff;
	temp_deriv_buff = activ_deriv_buff;
	activ_deriv_buff = temp_ptr;

	//hipMemcpy(activ_deriv_buff, temp_deriv_buff, max_activation_size * sizeof(float), hipMemcpyDeviceToDevice);

	printDeviceData("INIT CONV ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

	// BACKPROP OVER FIRST CONV LAYER

	// CONVOLUTION DIMENSIONS
	// hardcoded to 3 for the colors
	in_filters = 3;
	out_filters = dims -> init_conv_filters;
	in_spatial_dim = dims -> input;
	stride = dims -> init_conv_stride;
	kern_dim = dims -> init_kernel_dim;

	// CONVOLUTION FORWARD DATA
	conv_input = trainer -> cur_batch -> images;
	conv_weight = model_params -> init_conv_layer;
	// from backprop
	conv_out_deriv = activ_deriv_buff;

	// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
	// because residual
	conv_input_deriv = NULL;
	conv_weight_deriv = param_derivs -> init_conv_layer;

	prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, false, &data_algos[conv_algo_ind], &filt_algos[conv_algo_ind], &workspace_data[conv_algo_ind], &workspace_filt[conv_algo_ind]);
	conv_algo_ind++;

	printDeviceData("INIT CONV WEIGHT DERIV", conv_weight_deriv, print_size);

	hipFree(activ_deriv_buff);
	hipFree(temp_deriv_buff);
	hipFree(prev_conv_block_out_deriv);
	hipFree(block_activ_deriv);
}

void dump_parameters(int dump_id, Train_ResNet * trainer, const char * special_dir){

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;

	// values calculated from backprop, will reset these before returning
	Params * current_gradients = trainer -> backprop_buffer -> param_derivs;
	float ** current_gradient_locations = current_gradients -> locations;
	
	// running history values that the optimizer needs, will update these before returning
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	int param_size;
	float *model_location, *grad_location, * mean_location, * var_location;

	float * cpu_param_buff;
	FILE * fp;

	char * model_params_filepath;
	char * gradients_filepath;
	char * means_filepath;
	char * vars_filepath;

	int n_read, print_ret;
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		cpu_param_buff = (float *) malloc(param_size * sizeof(float));

		model_location = model_params_locations[i];
		hipMemcpy(cpu_param_buff, model_location, param_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&model_params_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/model_params/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(model_params_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(model_params_filepath);


		grad_location = current_gradient_locations[i];
		hipMemcpy(cpu_param_buff, grad_location, param_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&gradients_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/gradients/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(gradients_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(gradients_filepath);

		mean_location = prev_grad_means_locations[i];
		hipMemcpy(cpu_param_buff, mean_location, param_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&means_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/means/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(means_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(means_filepath);

		var_location = prev_grad_vars_locations[i];
		hipMemcpy(cpu_param_buff, var_location, param_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&vars_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/vars/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(vars_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(vars_filepath);

		free(cpu_param_buff);
	}
}


void dump_batch_norm_cache(Train_ResNet * trainer, char * filepath, Cache_BatchNorm * cache_batchnorm){

	FILE * fp;
	int n_wrote, print_ret;

	int input_size = cache_batchnorm -> input_size;
	int filters = cache_batchnorm -> feature_size;

	char * filepath_new = NULL;

	print_ret = asprintf(&filepath_new, "%smeans.buffer", filepath);
	float * cpu_means = (float *) malloc(filters * sizeof(float));
	hipMemcpy(cpu_means, cache_batchnorm -> means, filters * sizeof(float), hipMemcpyDeviceToHost);
	fp = fopen(filepath_new, "wb");
	n_wrote = fwrite(cpu_means, sizeof(float), filters, fp);
	fclose(fp);
	free(cpu_means);
	free(filepath_new);

	print_ret = asprintf(&filepath_new, "%sinv_vars.buffer", filepath);
	float * cpu_vars = (float *) malloc(filters * sizeof(float));
	hipMemcpy(cpu_vars, cache_batchnorm -> inv_vars, filters * sizeof(float), hipMemcpyDeviceToHost);
	fp = fopen(filepath_new, "wb");
	n_wrote = fwrite(cpu_vars, sizeof(float), filters, fp);
	fclose(fp);
	free(cpu_vars);
	free(filepath_new);
}

void dump_conv_block_activation(int dump_id, Train_ResNet * trainer, Activation_ConvBlock * activation_conv_block, int conv_block_ind, bool is_deriv, const char * special_dir){
	FILE * fp;
	int n_wrote, print_ret;

	char * filepath = NULL;

	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/conv_blocks/%02d/", special_dir, dump_id, conv_block_ind);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/conv_blocks/%02d/", special_dir, dump_id, conv_block_ind);
	}

	char * filepath_dup = NULL;
	
	char * batchnorm_filepath = NULL;
	if (is_deriv){
		print_ret = asprintf(&batchnorm_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/batch_norms/%02d/", special_dir, dump_id, conv_block_ind);
	}
	else{
		print_ret = asprintf(&batchnorm_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/batch_norms/%02d/", special_dir, dump_id, conv_block_ind);
	}

	char * batchnorm_filepath_dup = NULL; 

	int batch_size = trainer -> batch_size;
	int incoming_spatial_dim = activation_conv_block -> incoming_spatial_dim;
	int reduced_depth = activation_conv_block -> reduced_depth;
	int expanded_depth = activation_conv_block -> expanded_depth;
	int stride = activation_conv_block -> stride;


	/* REDUCTION CONV APPLIED */
	int reduction_size = incoming_spatial_dim * incoming_spatial_dim * reduced_depth * batch_size;
	float * cpu_reduction_applied = (float *) malloc(reduction_size * sizeof(float));
	hipMemcpy(cpu_reduction_applied, activation_conv_block -> post_reduced, reduction_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sreduction_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_reduction_applied, sizeof(float), reduction_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_reduction_applied);



	/* REDUCTION BATCH NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sreduced/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_reduced);
	free(batchnorm_filepath_dup);


	/* REDUCTION ACTIVATED */
	float * cpu_reduction_activated = (float *) malloc(reduction_size * sizeof(float));
	hipMemcpy(cpu_reduction_activated, activation_conv_block -> post_reduced_activated, reduction_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sreduction_activated.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_reduction_activated, sizeof(float), reduction_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_reduction_activated);


	/* SPATIAL CONV APPLIED */
	int spatial_size = incoming_spatial_dim * incoming_spatial_dim * reduced_depth * batch_size / (stride * stride);
	float * cpu_spatial_applied = (float *) malloc(spatial_size * sizeof(float));
	hipMemcpy(cpu_spatial_applied, activation_conv_block -> post_spatial, spatial_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sspatial_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_spatial_applied, sizeof(float), spatial_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_spatial_applied);


	/* SPATIAL BATCH NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sspatial/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_spatial);
	free(batchnorm_filepath_dup);


	/* SPATIAL ACTIVATED */
	float * cpu_spatial_activated = (float *) malloc(spatial_size * sizeof(float));
	hipMemcpy(cpu_spatial_activated, activation_conv_block -> post_spatial_activated, spatial_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sspatial_activated.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_spatial_activated, sizeof(float), spatial_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_spatial_activated);


	/* EXPANDED CONV APPLIED */
	int expanded_size = incoming_spatial_dim * incoming_spatial_dim * expanded_depth * batch_size / (stride * stride);
	float * cpu_expanded_applied = (float *) malloc(expanded_size * sizeof(float));
	hipMemcpy(cpu_expanded_applied, activation_conv_block -> post_expanded, expanded_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sexpanded_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_expanded_applied, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_expanded_applied);

	/* POST EXPANDED NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sexpanded/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_expanded);
	free(batchnorm_filepath_dup);


	/* (TRANSFORMED) RESIDUAL */

	// only blocks with projection weights haved a transformed residual. otherwise identity to input
	if (activation_conv_block -> transformed_residual) {
		float * cpu_residual = (float *) malloc(expanded_size * sizeof(float));
		hipMemcpy(cpu_residual, activation_conv_block -> transformed_residual, expanded_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&filepath_dup, "%stransformed_residual.buffer", filepath);
		fp = fopen(filepath_dup, "wb");
		n_wrote = fwrite(cpu_residual, sizeof(float), expanded_size, fp);
		fclose(fp);
		free(filepath_dup);
		free(cpu_residual);

		print_ret = asprintf(&batchnorm_filepath_dup, "%sprojected/", batchnorm_filepath);
		dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_projection);
		free(batchnorm_filepath_dup);

	}

	/* EXPANDED + RESIDUAL */
	float * cpu_combined_output = (float *) malloc(expanded_size * sizeof(float));
	hipMemcpy(cpu_combined_output, activation_conv_block -> output, expanded_size * sizeof(float), hipMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%scombined_output.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_combined_output, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_combined_output);


	free(filepath);
	free(batchnorm_filepath);

	

}

void dump_activations(int dump_id, Train_ResNet * trainer, Activations * activations, bool is_deriv, const char * special_dir){

	size_t batch_size = trainer -> batch_size;
	Dims * dims = trainer -> model -> dims;

	char * filepath = NULL;
	FILE * fp;
	int n_wrote, print_ret;

	// input
	size_t input_size = trainer -> cur_batch -> image_size * batch_size;
	if (!is_deriv){
		float * cpu_images = (float *) malloc(input_size * sizeof(float));
		hipMemcpy(cpu_images, trainer -> cur_batch -> images, input_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/input.buffer", special_dir, dump_id);
		fp = fopen(filepath, "wb");
		n_wrote = fwrite(cpu_images, sizeof(float), input_size, fp);
		fclose(fp);
		free(cpu_images);
		free(filepath);
	}


	/* 1. INIT CONV */

	size_t init_conv_applied_size = batch_size * dims -> init_conv_filters * (dims -> input / dims -> init_conv_stride) * (dims -> input / dims -> init_conv_stride);
	float * cpu_init_conv_applied = (float *) malloc(init_conv_applied_size * sizeof(float));
	hipMemcpy(cpu_init_conv_applied, activations -> init_conv_applied, init_conv_applied_size * sizeof(float), hipMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_conv_applied.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_conv_applied.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_conv_applied, sizeof(float), init_conv_applied_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_conv_applied);


	/* 2. INIT BATCH NORM */
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/batch_norms/init/", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/batch_norms/init/", special_dir, dump_id);
	}

	dump_batch_norm_cache(trainer, filepath, activations -> norm_init_conv);
	free(filepath);

	/* 3. ACTIVATED BATCH NORM */
	float * cpu_init_conv_activated = (float *) malloc(init_conv_applied_size * sizeof(float));
	hipMemcpy(cpu_init_conv_activated, activations -> init_conv_activated, init_conv_applied_size * sizeof(float), hipMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_conv_activated.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_conv_activated.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_conv_activated, sizeof(float), init_conv_applied_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_conv_activated);

	/* 4. MAX POOL */
	size_t maxpool_size = init_conv_applied_size / (dims -> init_maxpool_stride * dims -> init_maxpool_stride);

	float * cpu_init_convblock_input = (float *) malloc(maxpool_size * sizeof(float));
	hipMemcpy(cpu_init_convblock_input, activations -> init_convblock_input, maxpool_size * sizeof(float), hipMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_convblock_input.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_convblock_input.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_convblock_input, sizeof(float), maxpool_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_convblock_input);


	/* 5. CONV BLOCKS */
	int n_conv_blocks = activations -> n_conv_blocks;
	Activation_ConvBlock ** conv_blocks = activations -> activation_conv_blocks;
	Activation_ConvBlock * cur_conv_block;
	for (int i = 0; i < n_conv_blocks; i++){
		cur_conv_block = conv_blocks[i];
		dump_conv_block_activation(dump_id, trainer, cur_conv_block, i, is_deriv, special_dir);
	}


	/* 6. FINAL AVG POOL */
	int final_avg_pool_size = dims -> final_depth * batch_size;
	float * cpu_final_avg_pool = (float *) malloc(final_avg_pool_size * sizeof(float));
	hipMemcpy(cpu_final_avg_pool, activations -> final_conv_output_pooled, final_avg_pool_size * sizeof(float), hipMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/final_avg_pool.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/final_avg_pool.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_final_avg_pool, sizeof(float), final_avg_pool_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_final_avg_pool);

	/* 7. Fully Connected Output */
	int output_size = dims -> output * batch_size;
	float * cpu_linear_output = (float *) malloc(output_size * sizeof(float));
	hipMemcpy(cpu_linear_output, activations -> linear_output, output_size * sizeof(float), hipMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/fc_output.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/fc_output.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_linear_output, sizeof(float), output_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_linear_output);


	/* 8. Softmax Prediction */
	float * cpu_softmax = (float *) malloc(output_size * sizeof(float));
	if (is_deriv){
		hipMemcpy(cpu_softmax, trainer -> backprop_buffer -> output_layer_deriv, output_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/softmax.buffer", special_dir, dump_id);
	}
	else{
		hipMemcpy(cpu_softmax, trainer -> forward_buffer -> pred, output_size * sizeof(float), hipMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/softmax.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_softmax, sizeof(float), output_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_softmax);


	/* 9. Correct Classes */
	if (!is_deriv){
		int * correct_classes_cpu = trainer -> cur_batch -> correct_classes_cpu;
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/correct_classes.buffer", special_dir, dump_id);
		fp = fopen(filepath, "wb");
		n_wrote = fwrite(correct_classes_cpu, sizeof(int), batch_size, fp);
		free(filepath);
		fclose(fp);
	}
}

void dump_trainer_meta(int dump_id, Train_ResNet * trainer, const char * special_dir){

	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_metadata.txt", special_dir, dump_id);
	fp = fopen(filepath, "w");

	// DUMP THE BATCH INFO
	fprintf(fp, "%d\n", trainer -> batch_size);
	fprintf(fp, "%d\n", trainer -> cur_batch -> image_size);
	fprintf(fp, "%d\n", trainer -> cur_batch -> image_dim);
	fprintf(fp, "%d\n", trainer -> cur_batch -> shard_n_images);


	// NOW DO TRAINER METADATA
	fprintf(fp, "%f\n", trainer -> learning_rate);
	fprintf(fp, "%f\n", trainer -> weight_decay);
	fprintf(fp, "%f\n", trainer -> base_mean_decay);
	fprintf(fp, "%f\n", trainer -> base_var_decay);
	fprintf(fp, "%f\n", trainer -> cur_mean_decay);
	fprintf(fp, "%f\n", trainer -> cur_var_decay);
	fprintf(fp, "%f\n", trainer -> eps);
	fprintf(fp, "%d\n", trainer -> n_epochs);
	fprintf(fp, "%d\n", trainer -> cur_dump_id);
	fprintf(fp, "%d\n", trainer -> cur_epoch);

	for (int i = 0; i < trainer -> cur_epoch; i++){
		if (i == 0){
			fprintf(fp, "%f", (trainer -> loss_per_epoch)[i]);
		}
		else{
			fprintf(fp, ",%f", (trainer -> loss_per_epoch)[i]);
		}
	}
	fprintf(fp, "\n");

	for (int i = 0; i < trainer -> cur_epoch; i++){
		if (i == 0){
			fprintf(fp, "%f", (trainer -> accuracy_per_epoch)[i]);
		}
		else{
			fprintf(fp, ",%f", (trainer -> accuracy_per_epoch)[i]);
		}
	}
	fprintf(fp, "\n");

	fclose(fp);
}

void dump_trainer_checkpoint(int dump_id, Train_ResNet * trainer, const char * special_dir){

	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_checkpoint.txt", special_dir, dump_id);
	fp = fopen(filepath, "w");

	// DUMP THE BATCH INFO
	fprintf(fp, "%d\n", trainer -> cur_batch -> cur_shard_id);
	fprintf(fp, "%d\n", trainer -> cur_batch -> cur_batch_in_shard);

	// NOW DO TRAINER METADATA
	fprintf(fp, "%f\n", trainer -> cur_mean_decay);
	fprintf(fp, "%f\n", trainer -> cur_var_decay);
	fprintf(fp, "%d\n", trainer -> cur_dump_id);
	fprintf(fp, "%d\n", trainer -> cur_epoch);

	fclose(fp);
}

void dump_trainer(int dump_id, Train_ResNet * trainer, const char * special_dir){

	/* DUMP PARAMETERS */
	dump_parameters(dump_id, trainer, special_dir);
	
	/* DUMP FORWARD ACTIVATIONS */
	dump_activations(dump_id, trainer, trainer -> forward_buffer -> activations, false, special_dir);

	/* DUMP BACKPROP ACTIVATION DERIVS */
	/* NOT STORING THESE ANYMORE! */
	//dump_activations(dump_id, trainer, trainer -> backprop_buffer -> activation_derivs, true, special_dir);

	/* DUMP TRAINER METADATA */
	dump_trainer_meta(dump_id, trainer, special_dir);

	/* DUMP TRAINER CHECKPOINT */
	dump_trainer_checkpoint(dump_id, trainer, special_dir);

}

/* LOADING MODEL / ALLOCATING MEMORY FOR TRAINER */

// loading from a checkpoint that was dumped
// ASSUME EVERYTHING IS THE SAME AS IN THIS FILE EXCEPT: cur_shard_id, cur_batch_in_shard, cur_mean_decay, cur_var_decay, cur_dump_id, cur_epoch
void overwrite_trainer_hyperparams(Train_ResNet * trainer, int dump_id, const char * special_dir){

	// open the metadata file with hyper params and location of training sequence
	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_checkpoint.txt", special_dir, dump_id);
	fp = fopen(filepath, "r");

	// read the metadata file
	char * line;
	size_t len = 0;
	ssize_t n_read;

	// ASSUME THAT THE ORDERING OF LINES IS FIXED ACCORING TO "dump_trainer_checkpoint"

	// LOAD THE BATCH INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_batch -> cur_shard_id = atoi(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_batch -> cur_batch_in_shard = atoi(line);

	// LOAD OPTIMIZATION INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_mean_decay = atof(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_var_decay = atof(line);
	
	// LOAD SEQUENCE INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_dump_id = atoi(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_epoch = atoi(line);

	trainer -> init_loaded = 1;

	free(line);
	fclose(fp);
}


// LOADING THE MODEL PARAMS AND OPTIMIZATION STATES FROM CHECKPOINT
void overwrite_model_params(Train_ResNet * trainer, int dump_id, const char * special_dir){

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;
	
	// locations of optimization states
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	size_t param_size;
	float *model_location, * mean_location, * var_location;

	float * cpu_param_buff;
	FILE * fp;

	char * model_params_filepath;
	char * means_filepath;
	char * vars_filepath;

	int n_read, print_ret;
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		cpu_param_buff = (float *) malloc(param_size * sizeof(float));

		print_ret = asprintf(&model_params_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/model_params/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(model_params_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		model_location = model_params_locations[i];
		hipMemcpy(model_location, cpu_param_buff, param_size * sizeof(float), hipMemcpyHostToDevice);
		fclose(fp);
		free(model_params_filepath);

		print_ret = asprintf(&means_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/means/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(means_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		mean_location = prev_grad_means_locations[i];
		hipMemcpy(mean_location, cpu_param_buff, param_size * sizeof(float), hipMemcpyHostToDevice);
		fclose(fp);
		free(means_filepath);

		print_ret = asprintf(&vars_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/vars/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(vars_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		var_location = prev_grad_vars_locations[i];
		hipMemcpy(var_location, cpu_param_buff, param_size * sizeof(float), hipMemcpyHostToDevice);
		fclose(fp);
		free(vars_filepath);

		free(cpu_param_buff);
	}
}


// takes in pointers to GPU memory
void check_errors(Train_ResNet * trainer, int param_size, float * model_location, float * grad_location, float * mean_location, float * var_location, int location_ind){

	float * cpu_param_model = (float *) malloc(param_size * sizeof(float));
	hipMemcpy(cpu_param_model, model_location, param_size * sizeof(float), hipMemcpyDeviceToHost);

	float * cpu_param_grad = (float *) malloc(param_size * sizeof(float));
	hipMemcpy(cpu_param_grad, grad_location, param_size * sizeof(float), hipMemcpyDeviceToHost);

	float * cpu_param_mean = (float *) malloc(param_size * sizeof(float));
	hipMemcpy(cpu_param_mean, mean_location, param_size * sizeof(float), hipMemcpyDeviceToHost);

	float * cpu_param_var = (float *) malloc(param_size * sizeof(float));
	hipMemcpy(cpu_param_var, var_location, param_size * sizeof(float), hipMemcpyDeviceToHost);

	for (int i = 0; i < param_size; i++){
		if ((isnan(cpu_param_model[i])) || (isnan(cpu_param_grad[i])) || (isnan(cpu_param_mean[i])) || (isnan(cpu_param_var[i]))
				|| (isinf(cpu_param_model[i])) || (isinf(cpu_param_grad[i])) || (isinf(cpu_param_mean[i])) || (isinf(cpu_param_var[i]))){
			printf("ERROR: nan or inf found at location: %d\n", location_ind);
			printf("Dumping data to id=99999999 and exiting...\n");
			dump_trainer(99999999, trainer, trainer -> dump_dir);
			exit(1);
		}
	}

	free(cpu_param_model);
	free(cpu_param_grad);
	free(cpu_param_mean);
	free(cpu_param_var);
}



void * share_grad(void * thread_data_inp){

	Shared_Struct * shared_struct = ((struct thread_data *) thread_data_inp) -> shared_struct;
	int dev_id = ((struct thread_data *) thread_data_inp) -> dev_id;
	int n_devices = shared_struct -> n_devices;
	int n_locations = shared_struct -> n_locations;
	int location_ind = ((struct thread_data *) thread_data_inp) -> location_ind;
	int param_size = ((struct thread_data *) thread_data_inp) -> param_size;
	float * grad_location_local = ((struct thread_data *) thread_data_inp) -> grad_location_local;

	size_t total_param_floats = shared_struct -> total_param_floats;
	size_t * cum_sizes = shared_struct -> cum_sizes;
	float * all_shared_values = shared_struct -> values;
	size_t loc_in_shared = dev_id * total_param_floats + cum_sizes[location_ind]; 


	hipMemcpy(&all_shared_values[loc_in_shared], grad_location_local, param_size * sizeof(float), hipMemcpyDeviceToHost);

	shared_struct -> is_written[dev_id * n_locations + location_ind] = true;

}


void share_grads(Train_ResNet * trainer, int n_locations, int * param_sizes, float ** current_gradient_locations){

	pthread_t threads[n_locations];
	struct thread_data thread_data_array[n_locations];

	for (int i = 0; i < n_locations; i++){
		thread_data_array[i].dev_id = trainer -> id;
		thread_data_array[i].location_ind = i;
		thread_data_array[i].param_size = param_sizes[i];
		thread_data_array[i].grad_location_local = current_gradient_locations[i];
		thread_data_array[i].shared_struct = trainer -> shared_struct;
		pthread_create(&threads[i], NULL, share_grad, (void *) &thread_data_array[i]);
	}

	for (int i = 0; i < n_locations; i++){
			pthread_join(threads[i], NULL);
	}
}



void * get_reduced_grad(void * thread_data_inp){

	Shared_Struct * shared_struct = ((struct thread_data *) thread_data_inp) -> shared_struct;
	int dev_id = ((struct thread_data *) thread_data_inp) -> dev_id;
	int n_devices = shared_struct -> n_devices;
	int n_locations = shared_struct -> n_locations;
	int location_ind = ((struct thread_data *) thread_data_inp) -> location_ind;
	int param_size = ((struct thread_data *) thread_data_inp) -> param_size;
	float * grad_location_local = ((struct thread_data *) thread_data_inp) -> grad_location_local;

	size_t total_param_floats = shared_struct -> total_param_floats;
	size_t * cum_sizes = shared_struct -> cum_sizes;
	float * all_shared_values = shared_struct -> values;
	size_t loc_in_shared = dev_id * total_param_floats + cum_sizes[location_ind]; 


	while (!(shared_struct -> is_reduced[location_ind])) {
		// pass 
	}
	
	hipMemcpy(grad_location_local, &all_shared_values[location_ind], param_size * sizeof(float), hipMemcpyHostToDevice);
}


void get_reduced_grads(Train_ResNet * trainer, int n_locations, int * param_sizes, float ** current_gradient_locations){
	pthread_t threads[n_locations];
	struct thread_data thread_data_array[n_locations];

	for (int i = 0; i < n_locations; i++){
		thread_data_array[i].dev_id = trainer -> id;
		thread_data_array[i].location_ind = i;
		thread_data_array[i].param_size = param_sizes[i];
		thread_data_array[i].grad_location_local = current_gradient_locations[i];
		thread_data_array[i].shared_struct = trainer -> shared_struct;
		pthread_create(&threads[i], NULL, get_reduced_grad, (void *) &thread_data_array[i]);
	}

	for (int i = 0; i < n_locations; i++){
			pthread_join(threads[i], NULL);
	}
}



// doing ADAM optimizer
void update_parameters(Train_ResNet * trainer){
	
	size_t batch_size = (size_t) trainer -> batch_size;
	size_t image_size = (size_t) trainer -> cur_batch -> image_size;

	float learning_rate = trainer -> learning_rate;
	float weight_decay = trainer -> weight_decay;
	float base_mean_decay = trainer -> base_mean_decay;
	float base_var_decay = trainer -> base_var_decay;
	// update the running decays here...
	float cur_mean_decay = trainer -> cur_mean_decay * base_mean_decay;
	float cur_var_decay = trainer -> cur_var_decay * base_var_decay;
	float eps = trainer -> eps;

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;

	// values calculated from backprop, will reset these before returning
	Params * current_gradients = trainer -> backprop_buffer -> param_derivs;
	float ** current_gradient_locations = current_gradients -> locations;
	
	// running history values that the optimizer needs, will update these before returning
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	int param_size;
	float *model_location, *grad_location, * mean_location, * var_location;

	/* DUMP THE STATE OF TRAINING PROCESS! */
	// dumping every 10 batches, before update
	// also dump when nan or inf occurs (data dumped to id=99999999)
	int cur_dump_id = trainer -> cur_dump_id;

	if (cur_dump_id % 1000 == 0) {
		printf("DUMPING TRAINER @ ID: %d!\n\n", cur_dump_id);
		//dump_trainer(cur_dump_id, trainer, trainer -> dump_dir);
	}
	
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		model_location = model_params_locations[i];
		grad_location = current_gradient_locations[i];
		mean_location = prev_grad_means_locations[i];
		var_location = prev_grad_vars_locations[i];

		//check_errors(trainer, param_size, model_location, grad_location, mean_location, var_location, i);

		dim3 gridDimUpdate(ceil((float) param_size / MAX_THREAD_PER_BLOCK));
		dim3 blockDimUpdate(MAX_THREAD_PER_BLOCK);
		updateMeans <<< gridDimUpdate, blockDimUpdate >>> (param_size, grad_location, model_location, base_mean_decay, weight_decay, mean_location, i);
		updateVars <<< gridDimUpdate, blockDimUpdate >>> (param_size, grad_location, model_location, base_var_decay, weight_decay, var_location, i);
		updateParams <<< gridDimUpdate, blockDimUpdate >>> (param_size, model_location, mean_location, var_location, learning_rate, weight_decay, cur_mean_decay, cur_var_decay, eps, i);
	}


	

	/* RESET ALL VALUES TO 0 FOR NEXT PASS THROUGH BACKPROP */
	for (int i = 0; i < n_locations; i++){
		param_size = param_sizes[i];
		grad_location = current_gradient_locations[i];
		hipMemset(grad_location, 0, param_size * sizeof(float));
		// reset_forward_buffer(trainer);
		// reset_backward_buffer(trainer);
	}

	// reset images and classes before next hipMemcpy
	//hipMemset(trainer -> cur_batch -> images, 0, batch_size * image_size * sizeof(float));
	//hipMemset(trainer -> cur_batch -> correct_classes, 0, batch_size * sizeof(int));

	// change the current mean and var decay...
	trainer -> cur_mean_decay = cur_mean_decay;
	trainer -> cur_var_decay = cur_var_decay;
}


int main(int argc, char *argv[]) {

	int id = atoi(argv[1]);

	int shm_id = atoi(argv[2]);

	Shared_Struct * shared_struct = (Shared_Struct *) shmat(shm_id, NULL, 0);
	if (shared_struct == (void *) -1){
		printf("** * shmat error ***\n");
		exit(1);
	}

	int my_batch_size = atoi(argv[3]);

	int N_CLASSES = 1000;
	
	// GETTING CLASS METADETA
	char * LABEL_FILENAME = (char *) "../sample_data/id_to_label_mapping.txt";
	char * SYNSET_FILENAME = (char *) "../sample_data/id_to_synset_mapping.txt";
	char * COUNTS_FILENAME = (char *) "../sample_data/id_to_img_count_mapping.txt";
	Class_Metadata * class_metadata = populate_class_info(LABEL_FILENAME, SYNSET_FILENAME, COUNTS_FILENAME, N_CLASSES);
	int total_images = 0;
	for (int i = 0; i < N_CLASSES; i++){
		total_images += (class_metadata -> counts)[i];
	}

	// DEFINING MODEL DIMENSIONS
	int INPUT_DIM = 224;
	int INIT_KERNEL_DIM = 7;
	int INIT_CONV_FILTERS = 64;
	int INIT_CONV_STRIDE = 2;
	int INIT_MAXPOOL_DIM = 3;
	int INIT_MAXPOOL_STRIDE = 2;
	int N_CONV_BLOCKS = 16;
	int * IS_BLOCK_SPATIAL_REDUCTION = (int *) calloc(N_CONV_BLOCKS, sizeof(int));
	// transitions between spatial 56 -> 28 -> 14 -> 7
	// transitions between output depth of 256 -> 512 -> 1024 -> 2048
	int FINAL_DEPTH = 2048;
	IS_BLOCK_SPATIAL_REDUCTION[3] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[7] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[13] = 1;
	Dims * dims = init_dimensions(INPUT_DIM, INIT_KERNEL_DIM, INIT_CONV_FILTERS, INIT_CONV_STRIDE, INIT_MAXPOOL_DIM, INIT_MAXPOOL_STRIDE,
									N_CONV_BLOCKS, IS_BLOCK_SPATIAL_REDUCTION, FINAL_DEPTH, N_CLASSES);


	// declaring curandGenerator
	hiprandGenerator_t gen;
	// INITIALIZING RANDOM NUMBER GENERATOR USED TO INIT WEIGHTS
	hiprandStatus_t status_create = hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
	hiprandStatus_t status_set_seed = hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// INITIALIZING MODEL
	ResNet * model = init_resnet(dims, &gen);

	// INITIALIZING TRAINING

	// Batch Structure (will be modified every iteration of every epoch)
	
	// given when we generated shards...
	int SHARD_N_IMAGES = 32768;

	int BATCH_SIZE = 128;
	// dimensions of INPUT_DIM X INPUT_DIM x 3 color channels
	int IMAGE_SIZE = INPUT_DIM * INPUT_DIM * 3;
	Batch * batch = init_general_batch(BATCH_SIZE, IMAGE_SIZE, INPUT_DIM, SHARD_N_IMAGES);


	// General Training Structure (holds hyperparameters and pointers to structs which have network values)
	float LEARNING_RATE = 0.001;
	float WEIGHT_DECAY = 0;
	float MEAN_DECAY = 0.9;
	float VAR_DECAY = 0.999;
	float EPS = 0.0000001;
	float N_EPOCHS = 40;

	// INIT Cudnn
	miopenHandle_t miopen;
	miopenStatus_t miopen_status = miopenCreate(&miopen);
	//printf("Create Status: %s\n\n", miopenGetErrorString(cudnn_status));

	const char * MY_DUMP_DIR = "miopen_single";

	Train_ResNet * trainer = init_trainer(id, model, batch, shared_struct, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MEAN_DECAY, VAR_DECAY, EPS, N_EPOCHS, &miopen, MY_DUMP_DIR);

	// OVERRIDE IF LOADING WEIGHTS
	int LOAD_FROM_DUMP_ID = -1;

	
	
	if (LOAD_FROM_DUMP_ID != -1){
		overwrite_trainer_hyperparams(trainer, LOAD_FROM_DUMP_ID, MY_DUMP_DIR);
		overwrite_model_params(trainer, LOAD_FROM_DUMP_ID, MY_DUMP_DIR);
	}

	/* PERFORM TRAINING */


	int iterations_per_epoch = ceil((float) total_images / BATCH_SIZE);

	float *pred;
	int * correct;
	float epoch_n_wrong, batch_n_wrong;
	float epoch_loss, batch_loss, avg_batch_loss, epoch_accuracy, batch_accuracy, val_pred_correct;
	float total_images_per_epoch = BATCH_SIZE * iterations_per_epoch;

	int PRINT_FREQ = 1;

	hipError_t status;

	// if this was loaded from checkpoint
	int cur_epoch = trainer -> cur_epoch;
	int cur_iter_in_epoch = (trainer -> cur_dump_id + 1) % iterations_per_epoch;

	/* HARDCODE INFO BECAUSE WE ARE ONLY PROCESSING 2 SHARDS */
	N_EPOCHS = 1;
	int N_SHARDS = 2;
	iterations_per_epoch = (float) (N_SHARDS * SHARD_N_IMAGES) / BATCH_SIZE;

	struct timespec t1, t2;

	for (int epoch = cur_epoch; epoch < N_EPOCHS; epoch++){
		epoch_loss = 0;
		epoch_n_wrong = 0;
		for (int iter = cur_iter_in_epoch; iter < iterations_per_epoch; iter++){

			// start timer after first iteration because that is anomolous (computing best kernels on first pass)
			if (iter == 1){
				clock_gettime(CLOCK_REALTIME, &t1);
			}

			// if (iter == 10){
			// 	exit(0);
			// }
			//printf("************\n");

			/* LOAD NEW BATCH */
			//printf("Loading Batch...: %d\n", iter);
			// values go into trainer -> cur_batch -> [images_cpu|images_float_cpu|images|correct_classes_cpu|correct_classes]
			load_new_batch(trainer, class_metadata, trainer -> cur_batch);

			// hipDeviceSynchronize();
			// status = hipGetLastError();
			// printf("Status after loading batch: %s\n\n", hipGetErrorString(status));
			

			/* DO FORWARD PROP */
			// final predictions go into trainer -> forward_buffer -> [pred|pred_cpu|prediction_label]
			//printf("Making Predictions...\n");
			forward_pass(trainer);

			// hipDeviceSynchronize();
			// status = hipGetLastError();
			// printf("Status after forward pass: %s\n\n", hipGetErrorString(status));
			

			/* RECORD LOSS AND ACCURACY */
			if (iter % 1 == 0){
				hipDeviceSynchronize();

				// dimensions of pred: (BATCH_SIZE, N_CLASSES)
				pred = trainer -> forward_buffer -> pred_cpu;
				correct = trainer -> cur_batch -> correct_classes_cpu;
				
				// loss
				batch_loss = 0;
				for (int s = 0; s < BATCH_SIZE; s++){
					batch_loss += -1 * logf(pred[s * N_CLASSES + correct[s]]);
				}
				avg_batch_loss = batch_loss / BATCH_SIZE;
				epoch_loss += batch_loss;

				// accuracy
				batch_n_wrong = 0;
				for (int s = 0; s < BATCH_SIZE; s++){
					val_pred_correct = pred[s * N_CLASSES + correct[s]];
					for (int c = 0; c < N_CLASSES; c++){
						if ((c != correct[s]) && (pred[s * N_CLASSES + c] >= val_pred_correct)){
							batch_n_wrong++;
							break;
						}
					}
				}
				epoch_n_wrong += batch_n_wrong;
				batch_accuracy = 100 * ((float) BATCH_SIZE - batch_n_wrong) / ((float) BATCH_SIZE);


				if (iter % PRINT_FREQ == 0){
					printf("Epoch: %d, Batch: %d ----- Avg. Loss: %.4f, Accuracy: %.2f%%\n", epoch, iter, avg_batch_loss, batch_accuracy);
				}
				fflush(stdout);
			}


			/* DO BACKPROP */
			//printf("Backprop to Compute Derivs...\n");
			backwards_pass(trainer);

			// hipDeviceSynchronize();
			// status = hipGetLastError();
			// printf("Status after backwards pass: %s\n\n", hipGetErrorString(status));

			/* OPTIMIZE WEIGHTS */
			//printf("Applying Optimizer to Update Params...\n\n");
			update_parameters(trainer);

			// hipDeviceSynchronize();
			// status = hipGetLastError();
			// if (status != 0){
			// 	printf("Status after iter: %s\n\n", hipGetErrorString(status));
			// }
		}

		(trainer -> loss_per_epoch)[epoch] = epoch_loss;
		epoch_accuracy = (total_images_per_epoch - epoch_n_wrong) / total_images_per_epoch;
		(trainer -> accuracy_per_epoch)[epoch] = epoch_accuracy;
		printf("\nEpoch %d, Total Loss: %f\n", epoch, epoch_loss);
		printf("Epoch %d, Total Accuracy: %f\n\n", epoch, epoch_accuracy);
		fflush(stdout);

		// reset batch to start from beginning of dataset
		trainer -> cur_batch -> cur_shard_id = -1;
		trainer -> cur_batch -> cur_batch_in_shard = -1;

		trainer -> cur_epoch += 1;
		cur_iter_in_epoch = 0;

	}

	clock_gettime(CLOCK_REALTIME, &t2);

	double time = (t2.tv_sec - t1.tv_sec)  + (double) (t2.tv_nsec - t1.tv_nsec) / 1000000000.0;

	printf("\n\nTIME TAKEN: %1.31f\n", time);

	// DO A FINAL DUMP AFTER MODEL FINISHES (stored at 77777777)
	int FINAL_DUMP_ID = 77777777;
	//dump_trainer(FINAL_DUMP_ID, trainer, trainer -> dump_dir);

}