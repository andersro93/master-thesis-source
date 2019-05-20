#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <thread>
#include "gpu_kernel.cuh"
#include "tsetlin_random_wheel.cuh"

using namespace std;

GPUKernel::GPUKernel(){

    int available_gpus;

    // Query for amount of available GPUs
    if(cudaGetDeviceCount(&available_gpus) != cudaSuccess) {
        printf("KernelError: Unable to query for the amount of GPUs available! Check that the CUDA driver is properly installed.\n");
    }

    if(available_gpus == 0) {
        printf("KernelError: The CUDA driver could not find any compatible GPUs.\n");
    }

};

void GPUKernel::enable_gpu(unsigned int gpu_id) {

    // Try to select the correct GPU
    this->select_gpu(gpu_id);

    // Create a new configuration struct
    this->enabled_gpus.push_back(gpu_id);
}

void GPUKernel::remove_gpu(unsigned int gpu_id) {

    for(unsigned int index = 0; index < this->enabled_gpus.size(); index++) {

        if(this->enabled_gpus[index] == gpu_id) {
            this->enabled_gpus.erase(this->enabled_gpus.begin() + (index - 1));
            break;
        }
    }
}

void GPUKernel::load_model(const unsigned int* model, unsigned int classes, unsigned int clauses, unsigned int automatas, unsigned int states) {

    // Store the configuration to the kernel class
    this->classes_amount = classes;
    this->clauses_amount = clauses;
    this->automatas_amount = automatas;
    this->features_amount = static_cast<unsigned int>(automatas / 2);
    this->states_amount = states;

    // Check if an model has already been loaded, if so, deallocate its memory
    if(this->model != nullptr) {
        cudaFree(&this->model);
        this->model = nullptr;
    }

    // Attempt to allocate unified memory for the model
    if(cudaMallocManaged(&this->model, sizeof(unsigned int) * classes * clauses * automatas, cudaMemAttachGlobal) != cudaSuccess) {

        printf("load_model(): Unable to allocate unified memory for the model\n");
        return;
    }

    // Copy the data to GPU memory
    cudaMemcpy(this->model, model, sizeof(unsigned int) * classes * clauses * automatas, cudaMemcpyHostToHost); 

}

void GPUKernel::load_training_data(const unsigned int* x_train, const unsigned int* y_train, unsigned int train_samples_n) {

    // First cleanup if previous training data has been loaded
    if(this->x_train != nullptr) {
        cudaFree(&this->x_train);
        this->x_train = nullptr;
    }
    if(this->y_train != nullptr) {
        delete [] this->y_train;
        this->y_train = nullptr;
    }

    // Attempt to allocate unified memory for the x_data and y_data
    if(cudaMallocManaged(&this->x_train, sizeof(unsigned int) * train_samples_n * features_amount, cudaMemAttachGlobal) != cudaSuccess) {

        printf("load_training_data(): Unable to allocate unified memory for the x_train data\n");
        return;
    }

    // Allocate memory for y_train 
    this->y_train = new unsigned int[train_samples_n];

    // Copy the x train data to GPU memory
    cudaMemcpy(this->x_train, x_train, sizeof(unsigned int) * train_samples_n * features_amount, cudaMemcpyHostToHost);

    // Copy the y train data to Host memory
    memcpy(this->y_train, y_train, sizeof(unsigned int) * train_samples_n);

    // Store how many samples that have been loaded
    this->samples_train_n = train_samples_n;
}

void GPUKernel::load_validation_data(const unsigned int* x_val, const unsigned int* y_val, unsigned int val_samples_n) {

    // First cleanup if previous training data has been loaded
    if(this->x_val != nullptr) {
        cudaFree(&this->x_val);
        this->x_val = nullptr;
    }
    if(this->y_val != nullptr) {
        delete [] this->y_val;
        this->y_val = nullptr;
    }

    // Attempt to allocate unified memory for the x_data and y_data
    if(cudaMallocManaged(&this->x_val, sizeof(unsigned int) * val_samples_n * this->features_amount, cudaMemAttachGlobal) != cudaSuccess) {

        printf("load_validation_data(): Unable to allocate unified memory for the x_val data\n");
        return;
    }

    // Allocate memory for y_train 
    this->y_val = new unsigned int[val_samples_n];

    // Copy the x val data to GPU memory
    cudaMemcpy(this->x_val, x_val, sizeof(unsigned int) * val_samples_n * features_amount, cudaMemcpyHostToHost);

    // Copy the y val data to Host memory 
    memcpy(this->y_val, y_val, sizeof(unsigned int) * val_samples_n);

    // Store how many samples that have been loaded
    this->samples_val_n = val_samples_n;
}

void GPUKernel::fit(int epochs, int batches, bool validation, int threshold, float s, bool feedback, bool print_model_after_epoch)
{
    // Check if we have enabled any GPUs
    if(this->enabled_gpus.size() == 0)
    {
        printf("fit(): No GPUs has been enabled. Please enable some gpu's before trying to fit the data\n");
        return;
    }

    // Declare an array for the worker threads
    std::thread* worker_threads = new std::thread[this->classes_amount];
    
    // Create a new random generator
    TsetlinRandomWheel* random_generator = new TsetlinRandomWheel(rand(), this->classes_amount, 65565);

    float* training_times = new float[this->classes_amount];

    // Start looping the epochs
    for(int epoch = 1; epoch <= epochs; epoch++)
    {
        // Print feedback
        if(feedback == true){
            printf("Epoch %d \n", epoch);
        }

        // Start all the worker threads
        for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
            
            // Create a thread for each of the classes that will train and pass the parameters
            worker_threads[class_id] = std::thread(
                    &GPUKernel::train_class_one_epoch, 
                    class_id,
                    this->enabled_gpus[class_id % this->enabled_gpus.size()],
                    batches,
                    threshold,
                    s,
                    this->model,
                    this->x_train,
                    this->y_train,
                    this->samples_train_n,
                    this->classes_amount,
                    this->clauses_amount,
                    this->features_amount,
                    this->automatas_amount,
                    this->states_amount,
                    training_times,
                    random_generator
                    );
        }

        // Wait for all the threads to finish
        for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
            
            // Create a thread for each of the classes that will train and pass the parameters
            worker_threads[class_id].join();
        }

        // Check if we are to print the time for each class
        if(feedback) {

            printf("\nTraining time for classes: \n");

            for(unsigned int class_id = 0; class_id < classes_amount; class_id++) {
                printf("\t- Class %d: %f seconds\n", class_id, (training_times[class_id]/1000));
            }
            printf("\n");
        }

        // Check if we are to validate our model against the loaded validation data
        if(validation == true)
        {
            // If validation is turned on
            validate(feedback);
        }

        if(print_model_after_epoch == true) {
            
            print_model();
        }
    }

    // Some cleanup after the training is done
    delete [] worker_threads;
    delete [] training_times;
}

double GPUKernel::validate(bool feedback) {

    // Create some variables 
    double accuracy {0.0};
    unsigned int correct_guesses {0};
    unsigned int wrong_guesses {0};
    unsigned int* correct_guesses_for_class = new unsigned int[this->classes_amount] {0};
    unsigned int* wrong_guesses_for_class = new unsigned int[this->classes_amount] {0};
    unsigned int* total_predicted_for_class = new unsigned int[this->classes_amount] {0};
    unsigned int* total_samples_for_class = new unsigned int[this->classes_amount] {0};
    
    unsigned int correct_class;
    unsigned int temp_highest_class;
    int temp_highest_score;

    // GPU Arrays
    int* scores;
    cudaMallocManaged(&scores, sizeof(int) * this->classes_amount * this->samples_val_n, cudaMemAttachGlobal);
    cudaMemset(&scores, 0, sizeof(int) * this->classes_amount * this->samples_val_n);

    // Create an array that holds the threads that will validate each of the samples
    std::thread* worker_threads = new std::thread[this->classes_amount];

    // Start the validation for each class
    for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
        
        // Create and start the thread
        worker_threads[class_id] = std::thread(
            &GPUKernel::validate_class,
            class_id,
            this->enabled_gpus[class_id % this->enabled_gpus.size()],
            this->model,
            this->x_val,
            this->y_val,
            scores,
            this->samples_val_n,
            this->classes_amount,
            this->clauses_amount,
            this->features_amount,
            this->automatas_amount,
            this->states_amount
        );
    }

    // Wait for the threads to finish
    for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
        
        // Join the thread
        worker_threads[class_id].join();
    }

    // Calculate the output of for each sample
    for(unsigned int sample_id = 0; sample_id < this->samples_val_n; sample_id++) {
        
        // Just assign class 0 as the leading class
        temp_highest_class = 0;
        temp_highest_score = scores[(sample_id * this->classes_amount)];
        correct_class = this->y_val[sample_id];

        // Get the class with the most votes
        for(unsigned int class_id = 1; class_id < this->classes_amount; class_id++) {
            
            // Check if the current class has better score than previous
            if(temp_highest_score < scores[(sample_id * this->classes_amount) + class_id]) {
                temp_highest_score = scores[(sample_id * this->classes_amount) + class_id];
                temp_highest_class = class_id;
            }
        }

        // Check if we were correct
        if(temp_highest_class == correct_class) {
            
            // Store the correct guess to results
            correct_guesses += 1;
            correct_guesses_for_class[temp_highest_class] += 1;
        }
        else{
        
            // Store the wrong guess to results
            wrong_guesses += 1;
            wrong_guesses_for_class[temp_highest_class] += 1;
        }

        // Store how many times the class was predicted
        total_predicted_for_class[temp_highest_class] += 1;

        // Add how many guesses for that class that exists
        total_samples_for_class[correct_class] += 1;

        // printf("Guessed: %d, correct: %d, score: %d \n", temp_highest_class, correct_class, scores[(sample_id * this->classes_amount) + temp_highest_class]);
    }

    // Calculate the accuracy
    accuracy = (1.0 * correct_guesses) / (correct_guesses + wrong_guesses);

    // Check if we should print the results to console
    if(feedback == true) {
    
        // Print some info
        printf("Results from validation \n");
        printf("Total samples: %d \n", this->samples_val_n);
        printf("Model accuracy: %f \n", accuracy);
        printf("Correct guesses: %d \n", correct_guesses);
        printf("Wrong guesses: %d \n", wrong_guesses);

        for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
        
            printf("\n================ \n");
            printf("Class: %d \n", class_id);
            if(total_predicted_for_class[class_id] != 0) {
                printf("Precission: %f \n", (1.0 * correct_guesses_for_class[class_id]) / (total_predicted_for_class[class_id]));
            }
            else {
                printf("Precission: N/A \n");
            }
            if(total_samples_for_class[class_id] != 0) {
                printf("Recall: %f \n", (1.0 * correct_guesses_for_class[class_id]) / (total_samples_for_class[class_id]));
            }
            else {
                printf("Recall: N/A \n");
            }
            printf("Samples: %d \n", total_samples_for_class[class_id]);
            printf("Correct guesses: %d \n", correct_guesses_for_class[class_id]);
            printf("Wrong guesses: %d \n", wrong_guesses_for_class[class_id]);
        }

        printf("\n\n");
    
    }

    // Some cleanup after the validation is done
    delete [] correct_guesses_for_class;
    delete [] wrong_guesses_for_class;
    delete [] total_predicted_for_class;
    delete [] total_samples_for_class;
    delete [] worker_threads;
    cudaFree(&scores);

    return accuracy;
}

void GPUKernel::train_class_one_epoch(unsigned int class_id, unsigned int gpu_id, unsigned int batches, unsigned int threshold, float s, unsigned int* model, unsigned int* x_data, unsigned int* y_data, unsigned int samples, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int states_amount, float* training_times, TsetlinRandomWheel* random_generator) {

    // Attempt to select the given GPU
    if(cudaSetDevice(gpu_id) != cudaSuccess) {
        printf("train_class_one_epoch(): Unable to switch to gpu for Class: %d, GPU: %d \n", class_id, gpu_id);
        return;
    }

    // Create a stream for the class
    cudaStream_t class_stream;
    switch(cudaStreamCreateWithFlags(&class_stream, cudaStreamNonBlocking)) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidValue:
            printf("train_class_one(): An invalid value was passed to cudaStreamCreateWithFlags on Class %d, on GPU %d \n", class_id, gpu_id);
            break;
        default:
            printf("train_class_one(): An unknown CUDA error occured on cudaStreamCreateWithFlags with Class %d, on GPU %d, code: %d \n", class_id, gpu_id, cudaGetLastError());
    }

    // Allocate some memory that will be used during training
    bool* clauses_output;
    cudaMalloc(&clauses_output, sizeof(bool) * clauses_amount);

    int* score;
    cudaMalloc(&score, sizeof(int));

    unsigned int* clauses_feedback;
    cudaMallocManaged(&clauses_feedback, sizeof(unsigned int) * classes_amount);

    curandState* random_states;
    cudaMalloc(&random_states, sizeof(curandState) * clauses_amount * automatas_amount);

    // Declare some training specific variables
    bool correct_class;

    // Calculate the launch parameters for each kernel
    dim3 blocks = GPUKernel::calculate_blocks_per_kernel(clauses_amount);
    dim3 threads = GPUKernel::calculate_threads_per_block(automatas_amount);

    unsigned int reduce_votes_blocks = 1; // Due to the nature of the kernel, anyway its not like there are millions of clauses
    unsigned int reduce_votes_threads = (static_cast<unsigned int>(clauses_amount / 32) + 1) * 32;

    unsigned int calculate_feedback_blocks = 1; // This value will stay at one, unless we need more blocks
    unsigned int calculate_feedback_threads = (static_cast<unsigned int>(clauses_amount / 32) + 1) * 32;

    if(calculate_feedback_threads > 1024) {
        
        // Update the amount of blocks that are required
        calculate_feedback_blocks = (static_cast<unsigned int>(clauses_amount / calculate_feedback_threads)) + 1;

        // Set the amount of threads to be maximum
        calculate_feedback_threads = 1024;
    }

    // Check if we are above max threads per block
    if(reduce_votes_threads > 1024) {
        reduce_votes_threads = 1024;
    }

    // Initialize the random values
    initialize_random_states<<<blocks, threads, 0, class_stream>>>(random_states, rand(), clauses_amount * automatas_amount);

    // Create two events that will be used to measure total epoch training time
    cudaEvent_t start, stop;

    // Create the events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the time
    cudaEventRecord(start);

    // Start looping over the batches
    for(unsigned int batch_id = 0; batch_id < batches; batch_id++) {
    
        // Start looping over all of the samples
        for(unsigned int sample_id = 0; sample_id < samples; sample_id++) {
        
            // Check if the current class is the target class for this sample
            correct_class = (class_id == y_data[sample_id]);

            // Check if we are to train the sample on the current class or not
            if(correct_class || (random_generator->get_random_float(class_id)) < (1.0f / (1.0f * classes_amount))) {
                       
                // Evaluate the clause output
                validate_clauses<<<blocks, threads, 0, class_stream>>>(
                        model, 
                        clauses_output, 
                        x_data, 
                        sample_id, 
                        clauses_amount, 
                        features_amount, 
                        automatas_amount, 
                        class_id, 
                        states_amount, 
                        false
                );

                // Count the votes from the evaluation phase
                reduce_votes<<<reduce_votes_blocks, reduce_votes_threads, sizeof(int) * reduce_votes_threads, class_stream>>>(
                        score,
                        0,
                        clauses_output, 
                        clauses_amount, 
                        threshold
                );

                // Calculate the feedback to each clause
                calculate_feedback<<<calculate_feedback_blocks, calculate_feedback_threads, 0, class_stream>>>(
                        clauses_feedback,
                        score, 
                        threshold, 
                        s, 
                        class_id, 
                        correct_class, 
                        clauses_amount,
                        random_states
                );

                // Perform feedback on the model
                give_feedback_to_clauses<<<blocks, threads, 0, class_stream>>>(
                        model, 
                        clauses_feedback, 
                        x_data, 
                        clauses_output, 
                        class_id, 
                        sample_id, 
                        correct_class, 
                        clauses_amount, 
                        features_amount, 
                        automatas_amount, 
                        states_amount, 
                        threshold, 
                        s,
                        random_states
                );
            }
        }
    }

    // Set a stop timer
    cudaEventRecord(stop);

    // After launching all the kernels, try to wait for them to complete
    switch(cudaStreamSynchronize(class_stream)) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidResourceHandle:
            printf("train_class_one_epoch(): Unable to wait for stream before getting the score for Class %d, on GPU %d \n", class_id, gpu_id);
            break;
        default:
            printf("train_class_one_epoch(): An unknown CUDA error occured on cudaStreamSynchronize with Class %d, on GPU %d \n", class_id, gpu_id);
    }

    // Stop the time
    cudaEventSynchronize(stop);

    // Calculate the time difference
    cudaEventElapsedTime(&training_times[class_id], start, stop);

    // Free up space that was used during training
    cudaFree(clauses_output);
    cudaFree(score);
    cudaFree(clauses_feedback);
    cudaFree(random_states);

    // Attempt to destroy the stream
    if(cudaStreamDestroy(class_stream) != cudaSuccess) {
        printf("train_class_one_epoch(): Unable to destroy stream for Class %d, on GPU %d \n", class_id, gpu_id);
    }
}


void GPUKernel::validate_class(unsigned int class_id, unsigned int gpu_id, unsigned int* model, unsigned int* x_val, unsigned int* y_val, int* scores, unsigned int samples_amount, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state) {

    // Select the given GPU
    if(cudaSetDevice(gpu_id) != cudaSuccess) {
        printf("validate_class(): Unable to select GPU for Class: %d, GPU: %d \n", class_id, gpu_id);
        return;
    }

    // Create a stream for the thread
    cudaStream_t class_stream;
    switch(cudaStreamCreateWithFlags(&class_stream, cudaStreamNonBlocking)) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidValue:
            printf("validate_class(): An invalid value was passed to cudaStreamCreateWithFlags on Class %d, on GPU %d \n", class_id, gpu_id);
            break;
        default:
            printf("validate_class(): An unknown CUDA error occured on cudaStreamCreateWithFlags with Class %d, on GPU %d \n", class_id, gpu_id);
    }

    // Allocate memory for the clause outputs
    bool* clauses_output;
    cudaMalloc(&clauses_output, sizeof(bool) * clauses_amount);

    // Calculate some launch parameters
    dim3 blocks = GPUKernel::calculate_blocks_per_kernel(clauses_amount);
    dim3 threads = GPUKernel::calculate_threads_per_block(automatas_amount);

    // Start the validation of the samples
    for(unsigned int sample_id = 0; sample_id < samples_amount; sample_id++) {
        
        // Validate the sample
        validate_clauses<<<blocks, threads, 0, class_stream>>>(
                model, 
                clauses_output, 
                x_val, 
                sample_id, 
                clauses_amount, 
                features_amount, 
                automatas_amount, 
                class_id, 
                max_state,
                true
        );

        // Count the votes
        reduce_votes<<<1, 128, sizeof(int) * 128, class_stream>>>(
                scores,
                ((sample_id * classes_amount) + class_id),
                clauses_output, 
                clauses_amount, 
                0 
        );

    }

    // Wait for the stream to finish
    switch(cudaStreamSynchronize(class_stream)) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidResourceHandle:
            printf("validate_class(): Unable to wait for stream before getting the score for Class %d, on GPU %d", class_id, gpu_id);
            break;
        default:
            printf("validate_class(): An unknown CUDA error occured on cudaStreamSynchronize with Class %d, on GPU %d", class_id, gpu_id);
    } 

    // Cleanup used memory
    cudaFree(clauses_output);

    // Destroy the stream
    if(cudaStreamDestroy(class_stream) != cudaSuccess) {
        printf("validate_class(): Unable to destroy the stream for Class: %d, GPU %d \n", class_id, gpu_id);
        return;
    }
}

void GPUKernel::select_gpu(unsigned int gpu_id) {

    // Attempt to switch to the given GPU
    cudaError code = cudaSetDevice(gpu_id);

    if(code == cudaErrorDeviceAlreadyInUse) {
        printf("select_gpu(): Could not switch to the GPU with an ID of: %u, because it is already in use \n", gpu_id);
    }
    else if(code == cudaErrorInvalidDevice) {
        printf("select_gpu(): Could not switch to the GPU with an ID of: %u, because the GPU id does not exist in the CUDA driver\n", gpu_id);
    }
}

void GPUKernel::print_model() {

    printf("Model: \n");

    for(unsigned int class_id = 0; class_id < this->classes_amount; class_id ++) {
        printf("Class: %d \n", class_id);

        for(unsigned int clause_id = 0; clause_id < this->clauses_amount; clause_id ++) {
            printf("  %d: ", clause_id);

            for(unsigned int automata_id = 0; automata_id < this->automatas_amount; automata_id ++) {
                printf("%d ", this->model[(class_id * this->clauses_amount * this->automatas_amount) + (clause_id * this->automatas_amount) + automata_id]);
            }
            
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

dim3 GPUKernel::calculate_blocks_per_kernel(unsigned int clauses_amount) {

    return dim3(clauses_amount);
}

dim3 GPUKernel::calculate_threads_per_block(unsigned int automatas_amount) {

    if(automatas_amount > 1024) {
        automatas_amount = 1024;
    }
    else if(automatas_amount < 32) {
        automatas_amount = 32;
    }

    return dim3(automatas_amount);
}

GPUKernel::~GPUKernel() {


    // Free up memory from the devices
    if(this->model != nullptr) {
        cudaFree(&this->model);
    }

    if(this->x_train != nullptr) {
        cudaFree(&this->x_train);
    }
    
    if(this->x_val != nullptr) {
        cudaFree(&this->x_val);
    }
}
