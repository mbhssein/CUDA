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

#define CLIP(a) (a < 1U ? 1U : a > 255U ? 255U : a)
#define CUDA_CHECK(ERR) \
	if (ERR != cudaSuccess) { \
		cout << "Cuda error at line:" << __LINE__ << endl; \
		cout << cudaGetErrorString(ERR) << endl; \
    }
    

__global__ void Sobel(uint16_t* input, uint16_t* output, const int width, const int height, const int input_step, const int output_step)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_center = input_step * y + x;
    const int output_index = output_step * y + x;

    if ((x == 0) || (x == width - 1)
            || (y == 0) || (y == height - 1)) {

        output[output_index] = input[input_center];
        return;
    }

    const int SobelX[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int SobelY[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

    if ((x < width) && (y < height)) {
        int sumX = 0, sumY = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int data = input[input_center + i * input_step + j];
                sumX += data * SobelX[(i + 1) * 3 + j + 1];
                sumY += data * SobelY[(i + 1) * 3 + j + 1];
            }
        }
        output[output_index] = sqrt((double)(sumX * sumX + sumY * sumY) / 32);
    }
}

int main()
{
    /*
    Load frame pointer here
    uint16_t* rawFrame = 
    */ 
    int width = 1920 ; 
    int height = 1200 ; 

    long N = width * height;

    uint16_t* dL;
    CUDA_CHECK( cudaMallocManaged((void **) &dL, N * sizeof(uint16_t)));
    
    for(int pixel = 0; pixel < N ; pixel++)
    {
        dL[pixel] = (uint16_t)(rawFrame[pixel]);
    }

    const int input_step  = width * sizeof(uint16_t);
    const int output_step = width * sizeof(uint16_t);

    uint16_t* d_final_frame;
    cudaMallocManaged((void **) &d_final_frame, N * sizeof(uint16_t));
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const dim3 threadsPerBlock(64, 2);
    const dim3 blocksPerGrid((width + threadsPerBlock.x) / threadsPerBlock.x, (height + threadsPerBlock.y) / threadsPerBlock.y);

    Sobel<<<blocksPerGrid, threadsPerBlock>>>(dL, d_final_frame, width , height , input_step, output_step);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout <<"took " <<ms << " milliseconds" <<endl;

    ofstream myfile2 ("sobel.txt");
    for(long counter = 0; counter < N; counter++)
    {
        myfile2<< d_final_frame[counter]<<endl;
    }
    myfile2.close();

    cudaFree(dL);
    cudaFree(d_final_frame);
}
