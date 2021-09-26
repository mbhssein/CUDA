#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <array>
#include <chrono>
#include <string>

using namespace std::chrono;
using namespace std; 

#define TILE_WIDTH 1
#define TILE_HEIGHT  1
#define EDGE_VALUE_THRESHOLD  255
#define FILTER_DIAMETER  (2 * FILTER_RADIUS + 1)
#define FILTER_RADIUS 1

#define BLOCK_WIDTH	 (TILE_WIDTH + 2 * FILTER_RADIUS)
#define BLOCK_HEIGHT (TILE_HEIGHT + 2 * FILTER_RADIUS)

#define CLAMP_8bit(x) max(0, min(255, (x)))

#define CUDA_CHECK(ERR) \
	if (ERR != cudaSuccess) { \
		cout << "Cuda error at line:" << __LINE__ << endl; \
		cout << cudaGetErrorString(ERR) << endl; \
	}
  


__global__ void LaplacianFilter(float* g_DataIn, float* g_DataOut, int* width, int* height, float* d_LaplacianMatrix)
{
	__shared__ float sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
	{
		int index = y * (*width) + x;
		g_DataOut[index] = g_DataIn[index];

		return;
	}


	int index = y * (*width) + x;
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

	sharedMem[sharedIndex] = g_DataIn[index];

	__syncthreads();

	if(		threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
		&&	threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
	{
		float sum = 0;

		for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
			for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
			{
				float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
				sum += centerPixel * d_LaplacianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
			}

			//FIXME abs?
			float res = max(0.0, min((float)sum, 255.0));
			g_DataOut[index] = res;
	}	
}

int main()
{
	//Load data from ARF file
    /*
    Load frame pointer here
    uint16_t* rawFrame = 
    */ 

    int width = 1920 ; 
    int height = 1200 ; 

    const int FILTER_AREA	= FILTER_DIAMETER * FILTER_DIAMETER;

    int* d_Width = NULL;
    int* d_Height = NULL;

    cudaMalloc(&d_Width, sizeof(int));
    cudaMalloc(&d_Height, sizeof(int)); 
    CUDA_CHECK( cudaMemcpy(d_Width,  &width, sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_Height, &height, sizeof(int), cudaMemcpyHostToDevice) );

    long N = width * height;

    const float h_LaplacianMatrix[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	float* d_LaplacianMatrix;

    cudaMalloc((void**)&d_LaplacianMatrix, sizeof(float) * FILTER_AREA);
    cudaMemcpy(d_LaplacianMatrix,  h_LaplacianMatrix, sizeof(float) * FILTER_AREA, cudaMemcpyHostToDevice);

    float* dL;
    cudaMallocManaged((void **) &dL, N * sizeof(float));
 
    cudaMemset(dL, 0.0, N * sizeof(float));

    // int frameMin = *min_element(rawFrame, rawFrame + N); 
    // int frameMax = *max_element(rawFrame, rawFrame + N);

    for(int pixel = 0; pixel < N ; pixel++)
    {
        dL[pixel] = (float)(rawFrame[pixel]);
    }
    

    float* d_final_frame;
    cudaMallocManaged((void **) &d_final_frame, N * sizeof(float));

    // Time the calculation actions exception for read and write image.
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int gridWidth  = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int gridHeight = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    dim3 dimGrid(gridWidth, gridHeight);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    LaplacianFilter<<< dimGrid, dimBlock >>>(dL, d_final_frame, d_Width, d_Height, d_LaplacianMatrix);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << ms << endl;


    ofstream myfile2 ("lap.txt");
	for(long counter = 0; counter < N; counter++){
        myfile2<< d_final_frame[counter]<<endl;
    }
    myfile2.close();


    cudaFree(dL);
    cudaFree(d_final_frame); 

}
