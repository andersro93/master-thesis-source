#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "kernels.cuh"

__device__ __forceinline__ 
int get_polarity(int id) {

    // Check the id for a "for" polarity
    if ((id & 1) == 0) {

        // Id is a for polarity
        return 1;
    }

    // Previous check failed, therefore the id's polarity must be "against"
    return -1;
}

__device__ __forceinline__
bool automata_action(unsigned int automata_state, unsigned int max_state) {

    return (automata_state > (static_cast<unsigned int>(max_state / 2))); 
}

__device__ __forceinline__
int apply_threshold(int score, unsigned int threshold) {

    if(score > threshold) {
        score = static_cast<int>(threshold);
    }
    else if(score < -threshold) {
        score = -static_cast<int>(threshold);
    }

    return score;
}

__global__
void validate_clauses(unsigned int* model, bool* clauses_output, unsigned int* x_data, unsigned int sample_id, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int class_id, unsigned int max_state, bool prediction) 
{
    // Declare some shared variables
    // shared[0] = The output of the clause
    // shared[1] = Boolean flag if all is in exclude mode
    __shared__ bool shared[2];

    // Calculate the clause id to work on
    const int thread_id = threadIdx.x;

    // Initialize some "private variables"
    unsigned int sample_value;
    unsigned int automata_value;
    bool action;
    int automata_polarity;

    for (unsigned int clause_id = blockIdx.x; clause_id < clauses_amount; clause_id += gridDim.x) { 
    
        // Set the clause output to be true
        if(thread_id == 0) {
            shared[0] = true;
            shared[1] = true;
        }

        // Wait until all threads are ready
        __syncthreads();

        // Loop over each of the automata and "stride" through
        for(unsigned int automata_id = thread_id; automata_id < automatas_amount; automata_id += blockDim.x) {
        
            // Check if any of the other threads have evaluated the clause to false. This way we could skip checking.
            if(shared[0] == false) {
                break;
            }

            // Get the automatas value
            automata_value = model[(class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_id];

            // Get the action of the automata
            action = automata_action(automata_value, max_state);

            // Check if the automata is in an include state, if so, investigate further...
            if(action == true) {

                // Calculate the polarity of the automata
                automata_polarity = get_polarity(automata_id);

                // Get the sample's value
                sample_value = x_data[(sample_id * features_amount) + (automata_id / 2)]; 

                // Flip the flag that says that all automatas are in exclude mode
                shared[1] = false;

                // Since the automata is in an include state, lets check if the DOES NOT match the desired value
                if(((automata_polarity == 1) && (sample_value != 1)) || ((automata_polarity == -1) && (sample_value != 0))){
                
                    // A condition has been met that would falsify the entire clause. Therefore, evaluate the entire clause to false
                    shared[0] = false;
                    break;
                }
            }
        }

        // Wait until all threads to evaluate until finished
        __syncthreads();

        // Check if we are thread id 0
        if(thread_id == 0)
        {
            // Check if the clause was, when finished evaluating, evaluated to false
            if(shared[0] == false || (prediction == true && shared[1] == true)) {
                clauses_output[clause_id] = 0;
            }
            // Assuming it was not false, then it is true
            else {
                clauses_output[clause_id] = 1;
            }
        }
    }
}

__global__
void reduce_votes(int* scores, unsigned int scores_index, bool* clauses_output, unsigned int class_id, unsigned int clauses_amount, unsigned int threshold) {

    // Tempromary shared results
    extern __shared__ int results[];

    // Declare some private variables
    int thread_result = 0;

    for(unsigned int clause_id = threadIdx.x; clause_id < clauses_amount; clause_id += blockDim.x) {
        
        // Add the score to this threads tempromary score
        thread_result += (get_polarity(clause_id) * clauses_output[clause_id]);
    }

    // Move the threads result into shared memory
    results[threadIdx.x] = thread_result;

    // Wait until all the threads have completed the summation of all clause outputs
    __syncthreads();

    // Start to reduce the threads and score
    for(unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    
        // Check if this thread is doing some reduction
        if(threadIdx.x < offset) {
            results[threadIdx.x] += results[threadIdx.x + offset];
        }

        __syncthreads();
    }

    // Thread 0 will store the result in the scores list
    if(threadIdx.x == 0)
    {
        scores[scores_index] = results[threadIdx.x];
    }
}

__global__
void calculate_feedback(unsigned int* clauses_feedback, int* scores, unsigned int threshold, float s, unsigned int class_id, bool correct_class, unsigned int clauses_amount, curandState* random_states) {
    
    // Calculate the position of the thread
    unsigned int global_thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Declare some private variables
    curandState rnd_state = random_states[global_thread_id];
    int clause_polarity;
    int class_score = scores[0];

    // Loop all clauses 
    for (unsigned int clause_id = global_thread_id; clause_id < clauses_amount; clause_id += gridDim.x) {
   
        // Determine the polarity of the clause
        clause_polarity = get_polarity(clause_id);

        // Check if we are on the correct class
        if (correct_class == true) {
            
            // Check if we are to skip feedback for this clause
            if(curand_uniform(&rnd_state) > (((1.0 * threshold) - class_score) / (2.0 * threshold))) {

                // No feedback will be given to this clause
                clauses_feedback[clause_id] = 0;
                continue;
            }

            if(clause_polarity == 1) {
                
                clauses_feedback[clause_id] = 1;
            }
            else {
            
                clauses_feedback[clause_id] = 2;
            }
        }
        else {

            // Check if we are to skip feedback for this clause
            if(curand_uniform(&rnd_state) > (((1.0 * threshold) + class_score) / (2.0 * threshold))) {

                // No feedback will be given to this clause
                clauses_feedback[clause_id] = 0;
                continue;
            }

            if(clause_polarity == 1) {
                
                clauses_feedback[clause_id] = 2;
            }
            else {
            
                clauses_feedback[clause_id] = 1;
            }
        }
    }

    // Copy the random state back to global memory
    random_states[global_thread_id] = rnd_state;
}


__global__ 
void give_feedback_to_clauses(unsigned int* model, unsigned int* clauses_feedback, unsigned int* x_data, bool* clauses_output, unsigned int class_id, unsigned int sample_id, const bool correct_class, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state, unsigned int threshold, float s, curandState* random_states) {
    
    // Calculate and declare some "private variables"
    // Get the clause id, based on the block id in the grid
    unsigned int global_thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
   
    // Used to calculate the absolute index of an automata
    unsigned int automata_model_index;
    unsigned int automata_temp;

    // Used to tempromary store whether an automata is in include or exclude state 
    bool action;

    // Used to tempromary store the polarity of an automata
    int automata_polarity;

    // Used to tempromary store the feature id of which feature an automata is associated with
    unsigned int sample_value;

    // Get the random state from the random values matrix (used to generate "random" numbers)
    curandState rnd_state = random_states[global_thread_id];

    // In case there are more clauses than blocks, we need to loop them 
    for(unsigned int clause_id = blockIdx.x; clause_id < clauses_amount; clause_id += gridDim.x) {
        
        // Check if we are to do type 1 feedback
        if(clauses_feedback[clause_id] == 1){
        
            // If the clause output was evaluated to false
            if(clauses_output[clause_id] == 0) {
            
                // Loop and potentially punish all automatas
                for(unsigned int automata_index = threadIdx.x; automata_index < automatas_amount; automata_index += blockDim.x) {
                
                    // Calculate the position of the current automata
                    automata_model_index = (class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_index;

                    // Get the value for the automata
                    automata_temp = model[automata_model_index];

                    if((automata_temp > 1) && (curand_uniform(&rnd_state) <= (1.0 / s))) {
                        model[automata_model_index] = automata_temp - 1;
                    }
                }
            }
            else {
            
                // Loop over each of the automatas
                for(unsigned int automata_index = threadIdx.x; automata_index < automatas_amount; automata_index += blockDim.x){
                
                    // Calculate the position of the current automata
                    automata_model_index = (class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_index;
            
                    // Get the value of the sample for the current automata
                    sample_value = x_data[(sample_id * features_amount) + static_cast<unsigned int>(automata_index / 2)];

                    // Calculate the polarity of the automata
                    automata_polarity = get_polarity(automata_index);

                    // Get the value for the automata
                    automata_temp = model[automata_model_index];

                    // Check if the sample was False
                    if(sample_value == 0) {
                
                        // Check if the automata is an against automata
                        if(automata_polarity == -1){
                        
                            // Increment state
                            if((curand_uniform(&rnd_state) <= ((s - 1.0) / s)) && (automata_temp < max_state)) {
                                model[automata_model_index] = automata_temp + 1;
                            }
                        }
                        // Assumes that the automata is a for automata (since it is not an against automata)
                        else {

                            // Decrement state
                            if((curand_uniform(&rnd_state) <= (1.0 / s)) && automata_temp > 1) {
                                model[automata_model_index] = automata_temp - 1;
                            }
                        }

                    }
                    // Assumes that the sample is 1 (since it was not 0)
                    else {
                    
                        // Check if the automata is a for automata
                        if(automata_polarity == 1) {
                    
                            // Decrement the state 
                            if((curand_uniform(&rnd_state) <= ((s - 1.0) / s)) && (automata_temp < max_state)) {
                                model[automata_model_index] = automata_temp + 1;
                            }
                        }
                        // Assumes that the automata is an against automata (since it is not an for automata)
                        else {
                        
                            // Decrement state
                            if((curand_uniform(&rnd_state) <= (1.0 / s)) && automata_temp > 1) {
                                model[automata_model_index] = automata_temp - 1;
                            }
                        }
                    }
                }
            }
        }
        // Check if we are to do type 2 feedback
        else if(clauses_feedback[clause_id] == 2) {
        
            // Check if the clause was evaluated to true in the evaluation phase. 
            if(clauses_output[clause_id] == 1) {
            
                // Loop over all the automatas
                for(unsigned int automata_id = threadIdx.x; automata_id < automatas_amount; automata_id += blockDim.x) {
            
                    // Calculate the automata model index
                    automata_model_index = (class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_id;

                    // Get the automata value
                    automata_temp = model[automata_model_index];

                    // Get the sample's value
                    sample_value = x_data[(sample_id * features_amount) + (automata_id / 2)];

                    // Calculate the polarity of the automata
                    automata_polarity = get_polarity(automata_id);

                    // Get the include/exclude action for the automata
                    action = automata_action(automata_temp, max_state);

                    // Check if the automata is an for automata and that the feature is 0
                    if((automata_polarity == 1) && (sample_value == 0)){

                        // Check that the automata is in an exclude state and that we are not at max state
                        if((action == false) && (automata_temp < max_state)){
                        
                            model[automata_model_index] = automata_temp + 1;
                        }
                    }
                    else if((automata_polarity == -1) && (sample_value == 1)){

                        // Check that the automata is in an exclude state and that we are not at max state
                        if((action == false) && (automata_temp < max_state)){
                        
                            model[automata_model_index] = automata_temp + 1;
                        }
                    }
                }
            }
        }
    }

    // Some cleanup and persistence before exiting
    // Copy back the random state
    random_states[global_thread_id] = rnd_state;
}


__global__ 
void initialize_random_states(curandState* states, int seed, unsigned int amount_of_states) {

    // Calculate the global thread id
    unsigned int global_thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    // Calculate the offset (to make it a "bit more random")
    int offset = seed+global_thread_id;

    for(unsigned int index = global_thread_id; index < amount_of_states; index += gridDim.x) {
    
        // Initialize the random state
        curand_init(seed, index, offset, &states[index]);
    }
}
