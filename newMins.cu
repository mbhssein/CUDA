#include <iostream> 
#include <algorithm>    // std::min_element, std::max_element

using namespace std;

#define length 1200
#define width  1920
#define BINS 65336
#define BINS2 256
#define N (length * width)

#define CUDA_CHECK(ERR) \
	if (ERR != cudaSuccess) { \
		cout << "Cuda error at line:" << __LINE__ << endl; \
		cout << cudaGetErrorString(ERR) << endl; \
    }
    
__global__ void reduceSum(int* input, int* output)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    
    int* localIdx = input + (idx * BINS2); 

    long sum = 0; 
    if (tid < BINS2){
        sum += localIdx[tid];
        __syncthreads();
    }
    
    output[idx] = sum;
}

__global__ void scanMins(int* input, int* output)
{
    // int tid = threadIdx.x;
    // int idx = blockIdx.x; 
    int index = threadIdx.x + (blockIdx.x * blockDim.x );

    if (index > 0 && index < BINS2){
        if (input[index] <  input[index -1] && input[index] < input[index+1]){
            output[index] = index *256 ;
        }
    }
    

}


int main () {

    uint16_t* dImage = NULL;
	int *hist = NULL; 
    int *dHist = NULL;
    int * reducedHist = NULL;
    int * mins = NULL; 

    hist = (int*)malloc(BINS*sizeof(int));
    CUDA_CHECK(cudaMallocManaged(&dImage, length*width*sizeof(int)));
    /*
    Load frame pointer here
    uint16_t* rawFrame = 
    */ 

    int frameMin = *min_element(rawFrame, rawFrame + N); 
    int frameMax = *max_element(rawFrame, rawFrame + N);
    int range  = frameMax - frameMin; 

    cout<< frameMin<<" "<<frameMax<< endl;
    cout<< range<< endl;
    CUDA_CHECK(cudaMemcpy(dImage, rawFrame, length*width*sizeof(uint16_t), cudaMemcpyHostToDevice));
    
    // Create Histogram
    for(long x = 0; x < length*width; x++)
    {
        hist[rawFrame[x]]++;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int reducedSize = 256; 
    int reducedThreads = BINS/reducedSize; 
    /* place kernel calls below */ 

    CUDA_CHECK(cudaMallocManaged(&dHist, BINS*sizeof(int)));
    cudaMemcpy(dHist, hist, BINS * sizeof(int), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMallocManaged(&reducedHist, reducedSize*sizeof(int)));

    CUDA_CHECK(cudaMallocManaged(&mins, BINS2*sizeof(int)));

    reduceSum<<<reducedSize,reducedThreads>>>(dHist, reducedHist);
    CUDA_CHECK(cudaDeviceSynchronize());

    scanMins<<<16,16>>>(reducedHist, mins);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* end of kerenel calls */ 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << ms << endl;


    ofstream myfile1 ("hist.txt");
	for(int counter = 0; counter < BINS; counter++){
        myfile1<< dHist[counter]<<endl;
    }
    myfile1.close();

    ofstream myfile2 ("reduced.txt");
	for(int counter = 0; counter < BINS2; counter++){
        myfile2<< reducedHist[counter]<<endl;
    }
    myfile2.close();


    ofstream myfile3 ("mins.txt");
	for(int counter = 0; counter < BINS2; counter++){
        myfile3<< mins[counter]<<endl;
    }
    myfile3.close();


}
