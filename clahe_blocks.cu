#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <array>
#include <chrono>
#include <string>
#include <algorithm>

using namespace std::chrono;
using namespace std; 

#define BIN_SIZE 256 // Number of bins in the histogram
#define gridSize 128 // size of the grid for grid size of 16 we are working with 16 X 16 grid
#define width 1920 // frame width
#define height 1200 // frame height
#define rowBlocks ((height + gridSize -1 ) / gridSize) // number of blocks in the frame height ((height + gridSize -1 ) / gridSize)
#define colBlocks ((width + gridSize -1 ) / gridSize)  // number of blocks in the frame width ((width + gridSize -1 ) / gridSize)
#define threshold (gridSize * gridSize)// threshold of pixels to clip. Also depends on gridSize. e.g 16 grid size gives a total of 256 pizels in each tile. 
                      //the max value should not exceed 256. 100 is considered high.

#define CUDA_CHECK(ERR) \
	if (ERR != cudaSuccess) { \
		cout << "Cuda error at line:" << __LINE__ << endl; \
		cout << cudaGetErrorString(ERR) << endl; \
}

__global__ void localHistograms(float* L , int* Hists){
    
    int idx = threadIdx.x;
    int idy = blockIdx.x; 

    int rowStart = idx * gridSize;
    int colStart = idy * gridSize;
    // const int numRows = height / gridSize; 

    if ((rowStart + gridSize) > height-1){
        rowStart = height - gridSize -1;
    }

    if ((colStart + gridSize) > width -1){
        colStart = width - gridSize -1;
    }

    int pixel;
    for (int row = rowStart; row < rowStart + gridSize; row++) {
        for (int col = colStart; col < colStart + gridSize; col++) {
            pixel = (int)(L[col + row * width] * 255);
            // printf("%d \n", pixel);
            atomicAdd(&Hists[((idx + (idy * rowBlocks)) * BIN_SIZE) + pixel], 1);
        }
    }
}

__global__ void clipHistograms(int* dHist){
    int idx = threadIdx.x;
    int idy = blockIdx.x; 

    int index = idx + idy * blockDim.x; 
    int counter = 0;
    // if(index == 0) printf("checkpoint");
    int* hist = dHist + (index * BIN_SIZE);

    for(int i = 0; i < BIN_SIZE; i++){
        if (hist[i] > threshold ){
            counter = counter + hist[i] - threshold; 
            hist[i] = threshold;
        }
    }

    for(int j = 0; j < BIN_SIZE; j++){
        hist[j] = hist[j] + counter/BIN_SIZE + (j < counter%BIN_SIZE);
    }
}

__global__ void generateCDFs(float* dCdf, int* dHist){   
    int idx = threadIdx.x;
    int idy = blockIdx.x; 

    int index = idx + idy * blockDim.x; 
    int* hist = dHist + (index * BIN_SIZE);
    float* cdf = dCdf + (index * BIN_SIZE);
    cdf[0] = (float)hist[0];

    for (int i = 1 ; i < BIN_SIZE ; i++){
        cdf[i] = (cdf[i - 1] + hist[i]);  
    }

}

__global__ void transform(float* L , float* dCdf){
    
    int idx = threadIdx.x;
    int idy = blockIdx.x; 

    int rowStart = idx * gridSize;
    int colStart = idy * gridSize;
    // const int numRows = height / gridSize; 
    
    if ((rowStart + gridSize) > height -1 ){
        rowStart = height - gridSize -1;
        // printf("colStart = %d rowStart = %d \n",colStart,  rowStart);
    }

    if ((colStart + gridSize) > width-1){
        colStart = width - gridSize-1;
        // printf("rowStart = %d colStart = %d \n",rowStart, colStart);
    }


    int pixel;
    float newVal; 
    for (int row = rowStart; row < rowStart + gridSize; row++) {
        for (int col = colStart; col < colStart + gridSize; col++) {
            pixel = (int)(L[col + row * width] * 255);
            newVal = dCdf[((idx + (idy * rowBlocks)) * BIN_SIZE)+ pixel];
            L[col + row * width] = (float) newVal/255.f;
        }
    }
}


int main()
{

    /*
    Load frame pointer here
    uint16_t* rawFrame = 
    */ 

    long N = width * height;
    int num_block = rowBlocks * colBlocks;

    int frameMin = *min_element(rawFrame, rawFrame + N); 
    int frameMax = *max_element(rawFrame, rawFrame + N);
    cout<<frameMax<<" "<<frameMin<<endl;

    float* dL;
    cudaMallocManaged((void **) &dL, N * sizeof(float));
    cudaMemset(dL, 0.0, N * sizeof(float));

    // This loop to scale down the image pixel values between 0-1 then scale to 255 
    // the 27983 and 6444 are specific to this frame..
    // scalesPixel = (pixel - frameMin)/(frameMax - frameMin)

    for(int pixel = 0; pixel < N ; pixel++)
    {
        dL[pixel] = (float)((rawFrame[pixel] - frameMin)/(frameMax - frameMin));
        cout<< dL[pixel]<< endl;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(120, 75);

    int* dHist;
    CUDA_CHECK(cudaMallocManaged(&dHist, num_block * BIN_SIZE * sizeof(int)));
    cudaMemset(dHist, 0, num_block * BIN_SIZE * sizeof(int));

    float* dCdf;
    CUDA_CHECK(cudaMallocManaged(&dCdf, num_block * BIN_SIZE * sizeof(float)));
    cudaMemset(dCdf, 0.0, num_block * BIN_SIZE * sizeof(float));

    localHistograms<<<colBlocks , rowBlocks>>>(dL, dHist); // generate local histograms
    CUDA_CHECK(cudaDeviceSynchronize());

    clipHistograms<<<colBlocks , rowBlocks>>>(dHist); //clip histograms and distribute the clipped pixels to uniformly
    CUDA_CHECK(cudaDeviceSynchronize()); 

    generateCDFs<<<colBlocks,rowBlocks>>>(dCdf, dHist);// generate local cdfs
    CUDA_CHECK(cudaDeviceSynchronize()); 

    transform<<<colBlocks, rowBlocks>>>(dL, dCdf); // use cdfs to transform pixel values
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << ms << endl;

    ofstream myfile0 ("dHist.txt");
	for(long counter = 0; counter < num_block * BIN_SIZE; counter++){
        myfile0<< dHist[counter]<<endl;
    }
    myfile0.close();

    ofstream myfile2 ("CDFs.txt");
	for(long counter = 0; counter < num_block * BIN_SIZE; counter++){
        myfile2<< dCdf[counter]<<endl;
    }
    myfile2.close();

    ofstream myfile1 ("dL_processed-1.txt");
	for(long counter = 0; counter < width*height; counter++){
        myfile1<< dL[counter]<<endl;
    }
    myfile1.close();

    std::string command = "python plotImage.py "; //+ std::to_string(frameNum);
    cout<< command<<endl;
    system(command.c_str());

    cudaFree(dL);
    cudaFree(dHist);
    cudaFree(dCdf);
}
