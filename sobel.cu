#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include "/home/mohammed/projects/testingNPPs/arf_arf.C"
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

#define CUDA_CHECK(ERR) \
	if (ERR != cudaSuccess) { \
		cout << "Cuda error at line:" << __LINE__ << endl; \
		cout << cudaGetErrorString(ERR) << endl; \
	}

__global__ void SobelFilter(float* g_DataIn, float* g_DataOut, int* width, int* height, float* d_SobelMatrix)
{
    __shared__ float sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;

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
        float sumX = 0, sumY=0;

        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
            {
                float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
                sumX += centerPixel * d_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
                sumY += centerPixel * d_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
            }

        g_DataOut[index] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : (abs(sumX) + abs(sumY));
    }
}

int main()
{
	//Load data from ARF file
	arfobj arf_file_handle("/home/mohammed/frames-TM1531160339GR00-24/frames-TM1531160339GR00-24.arf");

	uint16_t* rawFrame = arf_file_handle.load16(1000);

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

    const float h_SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
	float* d_SobelMatrix;

    cudaMalloc((void**)&d_SobelMatrix, sizeof(float) * FILTER_AREA);
    cudaMemcpy(d_SobelMatrix,  h_SobelMatrix, sizeof(float) * FILTER_AREA, cudaMemcpyHostToDevice);

    float* dL;
    cudaMallocManaged((void **) &dL, N * sizeof(float));
    cudaMemset(dL, 0.0, N * sizeof(float));

    
    for(int pixel = 0; pixel < N ; pixel++)
    {
        dL[pixel] = (float)(rawFrame[pixel]);
    }
    cout<< dL[0]<<" "<< dL[N -1]<<endl;

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

    cout<< "gridWidth = " << gridWidth<< "  gridHeight = " <<gridHeight<< endl; 

    dim3 dimGrid(gridWidth, gridHeight);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    SobelFilter<<< dimGrid, dimBlock >>>(dL, d_final_frame, d_Width, d_Height, d_SobelMatrix);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << ms << endl;


    ofstream myfile2 ("sobel.txt");
	for(long counter = 0; counter < N; counter++){
        myfile2<< d_final_frame[counter]<<endl;
    }
    myfile2.close();


    cudaFree(dL);

}
