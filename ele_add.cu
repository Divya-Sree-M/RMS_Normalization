#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength * sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength * sizeof(float));
    cudaMallocManaged(&B, vectorLength * sizeof(float));
    cudaMallocManaged(&C, vectorLength * sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // --- NEW: CUDA EVENT TIMERS ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    // FIXED: Added parentheses to fix order of operations bug
    int blocks = (vectorLength + threads - 1) / threads;

    // --- NEW: Start recording right before the kernel launch ---
    cudaEventRecord(start, 0);

    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    
    // --- NEW: Stop recording right after the kernel finishes ---
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the stop event to be processed

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // Calculate execution duration
    // ------------------------------

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // --- NEW: THROUGHPUT CALCULATION ---
    // 1 float = 4 bytes. We read A (4B), read B (4B), and write C (4B) per element.
    double totalBytes = (double)vectorLength * sizeof(float) * 3; 
    double seconds = milliseconds / 1000.0;
    double throughputGBs = (totalBytes / 1e9) / seconds;

    printf("\n=== GPU PERFORMANCE RESULS ===\n");
    printf("Kernel Execution Time: %f ms (%f us)\n", milliseconds, milliseconds * 1000.0);
    printf("Effective Throughput:  %f GB/s\n\n", throughputGBs);

    // Clean Up Performance Metrics
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Clean Up Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}

//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}
extern "C" {
    // Returns the raw kernel execution time in milliseconds
    float runCudaVectorAdd(int vectorLength, float* hostA, float* hostB, float* gpuOutC) {
        float* A = nullptr;
        float* B = nullptr;
        float* C = nullptr;

        // Allocate Managed Memory
        cudaMallocManaged(&A, vectorLength * sizeof(float));
        cudaMallocManaged(&B, vectorLength * sizeof(float));
        cudaMallocManaged(&C, vectorLength * sizeof(float));

        // Copy source inputs to managed buffers
        memcpy(A, hostA, vectorLength * sizeof(float));
        memcpy(B, hostB, vectorLength * sizeof(float));

        // Create CUDA Performance Timers
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threads = 256;
        int blocks = (vectorLength + threads - 1) / threads;

        // Profile the raw execution block
        cudaEventRecord(start, 0);
        vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
        cudaEventRecord(stop, 0);
        
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy result back to Python's output pointer
        memcpy(gpuOutC, C, vectorLength * sizeof(float));

        // Cleanup resources
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        return milliseconds; // Pass execution time back to Python
    }
}
