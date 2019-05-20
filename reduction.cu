#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

// Definitions
#define CUDADEVICE 0
#define ARRAY_POWER_SIZE 30
#define RANDOM_SEED 1337
#define ENSEMBLES 50000


__global__
void generate_random_data(int* values, unsigned int values_n) {

    // Calculate global thread id
    unsigned int global_thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int stride = gridDim.x;

    // Initialize the random generator
    curandState_t random_state;
    curand_init(RANDOM_SEED, global_thread_id, pow(global_thread_id, 2), &random_state);

    // Start generating random numbers
    for(unsigned int index = global_thread_id; index < values_n; index += stride) {
    
        // Get a random value of either 0 or 1
        values[index] = (curand_uniform(&random_state) > 0.5) ? 1 : 0;
    }
}

__global__
void gpu_reduce(int* values, int* result, unsigned int amount) {

    // Declare some variables
    extern __shared__ int shared_memory[];

    // Declare some private variables
    int temp_score = 0;

    // Loop over all the samples
    for(unsigned int index = threadIdx.x; index < amount; index += blockDim.x) {
        
        // Store the score in the temp variable
        temp_score += values[index];
    }

    // Store the value in shared memory
    shared_memory[threadIdx.x] = temp_score * (1 - 2 * (threadIdx.x % 2));

    // Wait for all threads to finish the previous task
    __syncthreads();

    // Start to reduce the shared memory
    for(unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        
        // Check if we are inside the offset
        if(threadIdx.x < offset){
        
            // Add the offset value to the current value
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) {
        result[0] = shared_memory[0];
    }
}

void cpu_reduce(int* values, int* result, unsigned int amount) {

    // Declare a temp place to store the result (and initialize it to 0)
    int scores = 0;

    // Loop over the given amount of "clauses"
    for(unsigned int index = 0; index < amount; index++) {
        
        scores += (values[index] * (1 - 2 * (index % 2)));
    }

    // Store the result back to the given pointer
    result[0] = scores;
}

int main() {

    // Set the seed
    printf("Using random seed: %d \n", RANDOM_SEED);

    // Select the defined CUDA device
    printf("Using CUDA device: %d \n", CUDADEVICE);
    cudaSetDevice(CUDADEVICE);

    unsigned int array_size = pow(2, ARRAY_POWER_SIZE);

    // Declare the two arrays of data
    int* cpu_array;
    int* gpu_array;

    printf("Allocating memory \n");

    // Allocate memory in both RAM and VRAM
    cpu_array = (int*) malloc(sizeof(int) * array_size);
    cudaMalloc(&gpu_array, sizeof(int) * array_size);
    cudaMemset(gpu_array, 2, sizeof(int) * array_size);

    // Allocate the result variables
    int* cpu_result = (int*) malloc(sizeof(int));
    int* gpu_result;
    cudaMallocManaged(&gpu_result, sizeof(int), cudaMemAttachGlobal);

    // Allocate memory to store the results
    //double* cpu_results = (double*) malloc(sizeof(double) * ENSEMBLES * ARRAY_POWER_SIZE);
    //double* gpu_results = (double*) malloc(sizeof(double) * ENSEMBLES * ARRAY_POWER_SIZE);

    // Generating a random stream of data
    printf("Starting to create random values \n");
    generate_random_data<<<256, 1024>>>(gpu_array, array_size);
    cudaDeviceSynchronize();

    // Copy the data from RAM to VRAM
    printf("Copy random data from RAM to VRAM \n");
    cudaMemcpy(cpu_array, gpu_array, sizeof(int) * array_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(unsigned int index = 0; index < 65535; index++) {
        printf("%d", cpu_array[index]);
    }

    // Time variables
    clock_t cpu_start, cpu_stop;
    cudaEvent_t gpu_start, gpu_stop;

    float cpu_time, gpu_time;

    // The current dataset size
    int datasize;

    // Open a filestream to store the results
    std::ofstream result_file_cpu, result_file_gpu;
    result_file_cpu.open("reduce_cpu_results.csv");
    result_file_gpu.open("reduce_gpu_results.csv");


    // Lets create headers for each of the 2^n power experiments
    for(unsigned int power = 0; power <= ARRAY_POWER_SIZE; power++) {
        
        // Create the column header
        result_file_cpu << (int)pow(2, power);
        result_file_gpu << (int)pow(2, power);

        if(power != ARRAY_POWER_SIZE){
        
            result_file_cpu << ";";
            result_file_gpu << ";";
        }
    }

    // Print newline
    result_file_cpu << "\n";
    result_file_gpu << "\n";

    printf("Starting experiments...\n");
    // Run several ensembles in order to compensate for noise
    for(unsigned int ensemble = 0; ensemble < ENSEMBLES; ensemble++) {

        printf("Ensemble: %d \n", ensemble);
        
        // Start the experiments
        for(unsigned int power = 0; power <= ARRAY_POWER_SIZE; power++) {
    
            datasize = (int)pow(2, power);

            // Start the CPU time
            cpu_start = clock();

            // Perform the CPU reduction
            cpu_reduce(cpu_array, cpu_result, datasize);

            // Stop the CPU time
            cpu_stop = clock();

            // Store the CPU time in the time variable
            cpu_time = ((double)(cpu_stop - cpu_start)) / CLOCKS_PER_SEC;

            // Create the GPU time event objects
            cudaEventCreate(&gpu_start);
            cudaEventCreate(&gpu_stop);

            // Measure the time before launching the kernel
            cudaEventRecord(gpu_start);

            // Launch the GPU kernel
            gpu_reduce<<<1,1024,sizeof(int) * 1024>>>(gpu_array, gpu_result, datasize);

            // Measure the time after launching the kernel
            cudaEventRecord(gpu_stop);

            // Stop the time
            cudaEventSynchronize(gpu_stop);

            // Calculate the time difference
            cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);

            // Store result in result array
            // cpu_results[(ensemble * ARRAY_POWER_SIZE) + power] = (double)cpu_time;
            // gpu_results[(ensemble * ARRAY_POWER_SIZE) + power] = (double)gpu_time / 1000;
            result_file_cpu << (double)cpu_time;
            result_file_gpu << (double)gpu_time / 1000;

            if(power != ARRAY_POWER_SIZE){
                result_file_cpu << ";";
                result_file_gpu << ";";
            }
        }

        result_file_cpu << "\n";
        result_file_gpu << "\n";


    }

    // Write the output to a csv file
    // CPU time
    /*for(unsigned int ensemble_id = 0; ensemble_id < ENSEMBLES; ensemble_id++) {
        
        for(unsigned int power = 0; power <= ARRAY_POWER_SIZE; power++) {
        
            result_file_cpu << cpu_results[(ensemble_id * ARRAY_POWER_SIZE) + power] << ";";
        }

        result_file_cpu << "\n";
    }

    // GPU time
    for(unsigned int ensemble_id = 0; ensemble_id < ENSEMBLES; ensemble_id++) {
        
        for(unsigned int power = 0; power <= ARRAY_POWER_SIZE; power++) {
        
            result_file_gpu << gpu_results[(ensemble_id * ARRAY_POWER_SIZE) + power] << ";";
        }

        result_file_gpu << "\n";
    }
    */

    // Close the file streams
    result_file_cpu.close();
    result_file_gpu.close();

    // Deallocate results
    //free(cpu_results);
    //free(gpu_results);

    // Deallocate memory
    free(cpu_array);
    cudaFree(gpu_array);

    free(cpu_result);
    cudaFree(gpu_result);

    // Reset cuda device
    cudaDeviceReset();

    printf("Done! \n");

    return 0;
}
