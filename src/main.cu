#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>
#include <cuda_fp16.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_WIDTH 32

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};
using namespace std;

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};
__constant__ float ConvMask1[5 * 5 * 32];


static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

/**
  convolution layer-1 using tile + constant memory + layout transformation
  
  @param device_x: device x in global memory, which is input feature map
  @param device_a: device a in global memory, which is output feature map to return
 */
__global__ void ConvUnroll1Const(float* device_x, float* device_a) {
    __shared__ float  inputImg1[25][32];
    int thIdx = threadIdx.x;
    int thIdy = threadIdx.y;
    int imgIndex = blockIdx.x;
    
    // one block will be responsible for whole output image
    // so each block need to iterate 24 * 24 / 32 = 18 times, where (24 * 24) is output image size
    for (int i = 0; i < 18; i++) {
        float acc = 0.0f;
        // load image data 
        if (thIdy < 25) { // X_unroll out_H = 25
            int feaIdxY = thIdy / 5; // position in [5*5] mask
            int feaIdxX = thIdy % 5;
            int inputFeaStartPixelY = (i * 32 + thIdx) / 24; // top-left position in input image
            int inputFeaStartPixelX = (i * 32 + thIdx) % 24;;
            int inputIndex = imgIndex * 28 * 28 + (inputFeaStartPixelY + feaIdxY) * 28 + inputFeaStartPixelX + feaIdxX; // pixel index to be loaded by this thread
            inputImg1[thIdy][thIdx] = device_x[inputIndex];
        }
        __syncthreads();
        
        // matrix multiplication
        for (int j = 0; j < 25; j++) {
            float m1 = ConvMask1[j * 32 + thIdy];
            acc += m1 * inputImg1[j][thIdx];
        }
        
        // layout transformation: 24*24*32 -> 32*48*12
        int oriOutPixX = (i * 32 + thIdx) % 24; // original position in output map
        int oriOutPixY = (i * 32 + thIdx) / 24;
        int newPixX = oriOutPixX / 2; // new position in output map
        int newPixY = oriOutPixY / 2 * 4 + oriOutPixY % 2 * 2 + oriOutPixX % 2;
        // new 1-d index in output map
        int outIndex = imgIndex * 32 * 48 * 12 + thIdy * 48 * 12 + newPixY * 12 + newPixX;
        
        device_a[outIndex] = acc;
        __syncthreads();
    }
}

/**
  avgerage_pool layer-1 with layout transformation + relu4 embedded
  each block takes charge of 1 test image, thus 32 input feature map
  
  @param X: device X in global memory, which is input feature map for averaging
  @param Y: device Y in global memory, which is output feature map to return
 */
__global__ static void avgPoolParwithUnroll(float *X, float *Y) {
    // device_b changes from 24*24 to 48*12 after layout transformation
    int thIdxX = threadIdx.x;
    int thIdxY = threadIdx.y;
    for (const auto feaIndex : range(0, 32)) { // feature index
      float tmpSum = 0.0f;
      // iterate thru [2x2] sub-block
      for (const auto rowOffset : range(0, 2)) {
        for (const auto colOffset : range(0, 2)) {
            int xIndex = blockIdx.x * 32 * 48 * 12 + feaIndex * 48 * 12 + (thIdxY * 4 + rowOffset * 2 + colOffset) * 12 + thIdxX;
            
            // relu4 embedded
            float xVal = X[xIndex];
            if (xVal > 0) tmpSum += xVal;
        }
      }
      int yIndex = blockIdx.x * 32 * blockDim.x * blockDim.y + feaIndex * 12 * 12 + threadIdx.y * blockDim.x + threadIdx.x;
      // using multiplication instead of division
      Y[yIndex] = tmpSum * 0.25f;
    }
}

/**
  convolution layer-2 using tile + transposed tile (for mask) + layout transformation
  
  @param device_b: device b in global memory, which is input feature map for convolution
  @param device_conv2: device conv2 in global memory, which is the convolution mask
  @param device_c: device c in global memory, which is output feature map to return
 */
__global__ void ConvUnrollMemCoalDoubleHyp(float* device_b, float* device_conv2, float* device_c) {
    // mask matrix W' is [64, 5*5*32], so each tile needs to load 2 masks to cover 64 output feature maps
    __shared__ float  Mask1[32][32];
    __shared__ float  Mask2[32][32];
    // (H_out * W_out) of output feature map is 64, so each tile also needs to load 2 pixel values
    __shared__ float  inputImg1[32][32];
    __shared__ float  inputImg2[32][32];
    // therefore each thread will be responsible for 4 output pixels
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    int thIdx = threadIdx.x;
    int thIdy = threadIdx.y;
    int imgIndex = blockIdx.x;
    
    // X_unroll W_out is (5*5*64), each tile will load 64 in each iteration, so we need 25 iteration
    for (int i = 0; i < 25; i++) {
        // load mask values (transposed here)
        int maskOutIdx = thIdx; // output feature index (0 to 31)
        int maskInIdx = (i * 32 + thIdy) / 25; // input feature index (0 to 31)
        int maskPixIdx = (i * 32 + thIdy) % 25; // mask pixel index (0 to 24)
        int offset = (maskPixIdx * 32 + maskInIdx) * 64;
        Mask1[thIdx][thIdy] = device_conv2[offset + maskOutIdx]; // transposed for memory coalescing
        Mask2[thIdx][thIdy] = device_conv2[offset + maskOutIdx + 32];
        
        // load input image values
        int inputFeaIdx = (i * 32 + thIdy) / 25; // input feature index
        int inputFeaBlockPixel = (i * 32 + thIdy) % 25;
        int feaIdxY = inputFeaBlockPixel / 5; // pixel index inside mask
        int feaIdxX = inputFeaBlockPixel % 5;
        int inputFeaStartPixelY = thIdx / 8; // top-left pixel of mask
        int inputFeaStartPixelX = thIdx % 8;
        
        // layout transformation
        // input map changes from [12*12*32] to [32*12*12]
        int inputIndex1 = imgIndex * 12 * 12 * 32 + inputFeaIdx * 12 * 12 + (inputFeaStartPixelY + feaIdxY) * 12 + inputFeaStartPixelX + feaIdxX;
        int inputIndex2 = imgIndex * 12 * 12 * 32 + inputFeaIdx * 12 * 12 + (inputFeaStartPixelY + feaIdxY + 4) * 12 + inputFeaStartPixelX + feaIdxX;
        
        inputImg1[thIdy][thIdx] = device_b[inputIndex1];
        inputImg2[thIdy][thIdx] = device_b[inputIndex2];
        __syncthreads();
        
        for (int j = 0; j < 32; j++) {
            acc11 += Mask1[thIdy][j] * inputImg1[j][thIdx];
            acc12 += Mask2[thIdy][j] * inputImg1[j][thIdx];
            acc21 += Mask1[thIdy][j] * inputImg2[j][thIdx];
            acc22 += Mask2[thIdy][j] * inputImg2[j][thIdx];
        }
        __syncthreads();
    }
    
    // output changes from 8*8*64 to 64*16*4
    int oriOutPixX = thIdx % 8; // original output pixel
    int oriOutPixY = thIdx / 8;
    int newPixX = oriOutPixX / 2; // new output pixel
    int newPixY = oriOutPixY / 2 * 4 + oriOutPixY % 2 * 2 + oriOutPixX % 2;
    int outIndex = imgIndex * 64 * 16 * 4 + thIdy * 16 * 4 + newPixY * 4 + newPixX;
    device_c[outIndex] = acc11;
    device_c[outIndex + 32 * 64] = acc12;
    int outIndex2 = imgIndex * 64 * 16 * 4 + thIdy * 16 * 4 + newPixY * 4 + newPixX + 4 * 8;
    device_c[outIndex2] = acc21;
    device_c[outIndex2 + 32 * 64] = acc22;
}

/**
  avgerage_pool layer-2 with layout transformation + relu4 embedded
  each block takes charge of 1 test image, thus 64 input feature map
  
  @param X: device X in global memory, which is input feature map for averaging
  @param Y: device Y in global memory, which is output feature map to return
 */
__global__ static void avgPoolParwithUnroll2(float *X, float *Y) {
    // device_b changes from 8*8 to 16*4
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // output pixel
    int thIdxX = threadIdx.x;
    int thIdxY = threadIdx.y;
    for (const auto feaIndex : range(0, 64)) { // feature index (0 to 63)
      float tmpSum = 0.0f;
      for (const auto rowOffset : range(0, 2)) {
        for (const auto colOffset : range(0, 2)) {
            int xIndex = blockIdx.x * 16 * 4 * 64 + feaIndex * 16 * 4 + (thIdxY * 4 + rowOffset * 2 + colOffset) * 4 + thIdxX;
            
            // relu4 embedded
            float xVal = X[xIndex];
            if (xVal > 0) tmpSum += xVal;
        }
      }
      // output transformed from [8*8*64] to [64*8*8]
      Y[index * 64 + feaIndex] = tmpSum * 0.25f;
    }
}

/**
  relu2 in parallel
  
  @param X: device X in global memory, which is input feature map to relu
  @param inputLength: input length value
 */
__global__ static void relu2Parallel(float *X, int inputLength) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < inputLength) {
        X[i] = (X[i] < 0) ? 0 : X[i];
    }
}

/**
  Choose the guess with largest score (which is a host function)
  
  @param X: device X in host memory, which is input feature map for finding the max value
  @param xdims[2]: input length value
  @param Y: image labels value to return
 */
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

/**
  argmax in parallel
  
  @param X: device X in global memory, which is input feature map to argMax
  @param xdims0: input image set size
  @param xdims1: 10
  @param Y: device Y in global memory, which is output image label array to return
 */
__global__ static void argMaxParallel(float *X, int xdims0, int xdims1, int *Y) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < xdims0) { // boundary check
      int max_idx = 0;
      float max     = X[i * xdims1];
      for (const auto j : range(0, xdims1)) {
        float elem = X[(i * xdims1) + j];
        if (elem > max) { // update max
          max_idx = j;
          max     = elem;
      }
    }
    Y[i] = max_idx;
    }
}

/**
  fully_forward with tile implementation
  
  @param X: device X in global memory, which is first matrix in multiplication
  @param numXRows: number of rows in X
  @param numXColumns: number of columns in X
  @param W: device W in global memory, which is second matrix in multiplication
  @param numWRows: number of rows in W
  @param numWColumns: number of columns in W
  @param Y: device Y in global memory, which is the result matrix to return
  @param numYRows: number of rows in Y
  @param numYColumns: number of columns in Y
 */
__global__ void fully_forward_tile(const float *X, const int numXRows, const int numXColumns, const float *W, const int numWRows, const int numWColumns, float *Y, const int numYRows, const int numYColumns) {

  //initialize shared memory
  __shared__ float Xds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Wds[TILE_WIDTH][TILE_WIDTH];
  //initialize block and thread index
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x;  int ty = threadIdx.y;

  //initialize row and col
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float sum = 0.0f;
  int offset = 0;
  if (numXColumns % TILE_WIDTH)
  {
    offset = 1;
  }
  for(int ph = 0; ph < (numXColumns / TILE_WIDTH) + offset; ++ph){

    Xds[ty][tx] = X[Row * numXColumns + ph * TILE_WIDTH + tx];
    Wds[ty][tx] = W[(ph * TILE_WIDTH + ty) * numWColumns + Col];
    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k){

      if (
        (Row < numXRows) &&
        ((ph * TILE_WIDTH + k) < numXColumns) &&
        ((ph * TILE_WIDTH + k) < numWRows) &&
        (Col < numWColumns)
      )
      sum += Xds[ty][k] * Wds[k][tx];
    }
    __syncthreads();

    if((Row < numXRows) && (Col < numWColumns))
    Y[Row * numWColumns + Col] = sum;
  }
}

// Forward operation for the CNN, a combination of convolution + average_pool
// + relu + fully_forward
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out, 
                      float *device_x, float *device_a, float *device_b, float *device_c, float *device_d, float *device_e, float *device_f, float *device_conv2, float *device_fc1, float *device_fc2, int *deviceOutput) {
  // convolution layer 1
  int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), 
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  int size_x = xdims[0] * xdims[1] * xdims[2] * xdims[3];
  int size_conv1 = conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
    
  cudaMemcpy(device_x, x, size_x*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ConvMask1, conv1, size_conv1 * sizeof(float));
  
  dim3 gridDimConv1(xdims[0], 1, 1);
  dim3 blockDimConv1(TILE_WIDTH, TILE_WIDTH, 1); // [32, 32]
  ConvUnroll1Const<<<gridDimConv1, blockDimConv1>>>(device_x, device_a);
  cudaDeviceSynchronize();
  
  cudaFree(device_x); 

  // relu + average_pool layer 1
  int bdims[]   = {adims[0], adims[1] / 2, adims[2] / 2,
                       adims[3]};
    
  //cudaMalloc((void **) &device_b, inputLengthB * sizeof(float));
  dim3 dimGridAvg1(bdims[0], 1, 1); // batch size k
  dim3 dimBlockAvg1(bdims[2], bdims[1], 1); // output width * height, [12, 12]
  avgPoolParwithUnroll<<<dimGridAvg1, dimBlockAvg1>>>(device_a, device_b);
  cudaDeviceSynchronize();
    
  cudaFree(device_a);
   
  // convolution layer 2
  int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), 
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  int size_conv2=conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
  cudaMemcpy(device_conv2, conv2, size_conv2*sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridDimConv2(bdims[0], 1, 1);
  dim3 blockDimConv2(TILE_WIDTH, TILE_WIDTH, 1); // [32, 32]
  ConvUnrollMemCoalDoubleHyp<<<gridDimConv2, blockDimConv2>>>(device_b, device_conv2, device_c);
  cudaDeviceSynchronize();
 
  cudaFree(device_b);
  cudaFree(device_conv2); 
    
  // relu + average_pool layer 2
  int ddims[] = {cdims[0], cdims[1] / 2, cdims[2] / 2, cdims[3]};
    
  dim3 dimGridAvg2(ddims[0], 1, 1);
  dim3 dimBlockAvg2(ddims[2], ddims[1], 1); 
  avgPoolParwithUnroll2<<<dimGridAvg2, dimBlockAvg2>>>(device_c, device_d);
  cudaDeviceSynchronize();
    
  cudaFree(device_c);
      
  // fully_forward layer 1
  int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
  int edims[] = {ddims[0], fc1dims[1]};
  
  cudaMemcpy(device_fc1, fc1,(fc1dims[0] * fc1dims[1]) * sizeof(float), cudaMemcpyHostToDevice );
  dim3 DimGrid(ceil(fc1dims[1] / 32.0), ceil(ddims2[0] / 32.0),1);
  dim3 DimBlock(32, 32, 1);
  fully_forward_tile<<<DimGrid,DimBlock>>>(device_d, ddims2[0], ddims2[1], device_fc1, fc1dims[0], fc1dims[1], device_e, ddims2[0], fc1dims[1]);
  cudaDeviceSynchronize();

  cudaFree(device_d);
  cudaFree(device_fc1);

  // relu2 
  int inputLengthE = edims[0] * edims[1];
    
  relu2Parallel<<<ceil(inputLengthE/1024.0), 1024>>>(device_e, inputLengthE);
  cudaDeviceSynchronize();

  // fully_forward layer 2
  int fdims[] = {edims[0], fc2dims[1]};
    
  cudaMemcpy(device_fc2, fc2,(fc2dims[0] * fc2dims[1]) * sizeof(float), cudaMemcpyHostToDevice );

  fully_forward_tile<<<DimGrid,DimBlock>>>(device_e, edims[0], edims[1], device_fc2, fc2dims[0], fc2dims[1], device_f, fdims[0], fdims[1]);
  cudaDeviceSynchronize();
    
  cudaFree(device_e);
  cudaFree(device_fc2);

  // arg_max
  int inputLengthO = fdims[0];
    
  argMaxParallel<<<ceil(inputLengthO/1024.0), 1024>>>(device_f, fdims[0], fdims[1], deviceOutput);
  cudaDeviceSynchronize();
    
  cudaMemcpy(out, deviceOutput, inputLengthO * sizeof(int), cudaMemcpyDeviceToHost);
    
  cudaFree(device_f); 
  cudaFree(deviceOutput);
}

int main(int argc, char **argv) {
                                        
  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);
    
  // mallocate memory
  float *device_x;
  float *device_a;
  float *device_b;
  float *device_conv2;
  float *device_c;
  float *device_d;
  float *device_fc1;
  float *device_e;
  float *device_f;
  float *device_fc2;
  int *deviceOutput;

  int size_x=xdims[0]*28*28;
  int size_a=xdims[0]*24*24*32;
  cudaMalloc((void**)&device_x, size_x*sizeof(float));
  cudaMalloc((void**)&device_a, size_a*sizeof(float));

  int inputLengthB =xdims[0]*12*12*32;
  cudaMalloc((void **) &device_b, inputLengthB * sizeof(float));

  int size_c=xdims[0]*8*8*64;
  int size_conv2=5*5*32*64;
  cudaMalloc((void**)&device_c, size_c*sizeof(float));
  cudaMalloc((void**)&device_conv2, size_conv2*sizeof(float));

  int inputLengthD =xdims[0]*4*4*64;
  cudaMalloc((void **) &device_d, inputLengthD * sizeof(float));

  cudaMalloc((void**) &device_fc1, (1024*128) * sizeof(float));
  cudaMalloc((void**) &device_e, (xdims[0] * 128) * sizeof(float));

  cudaMalloc((void**) &device_fc2, (128 * 10) * sizeof(float));
  cudaMalloc((void**) &device_f, (xdims[0] * 10) * sizeof(float));

  cudaMalloc((void **) &deviceOutput, xdims[0] * sizeof(int));
    
  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out, device_x, device_a, device_b, device_c, device_d, device_e, device_f, device_conv2, device_fc1, device_fc2, deviceOutput);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}