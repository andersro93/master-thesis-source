#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ __forceinline__
int get_polarity(int id);

__device__ __forceinline__
bool automata_action(unsigned int automata_state, unsigned int max_state);

__global__
void reduce_votes(int* scores, unsigned int scores_index, bool* clauses_output, unsigned int clauses_amount, unsigned int threshold);

__global__
void validate_clauses(unsigned int* model, bool* clauses_output, unsigned int* x_data, unsigned int sample_id, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int class_id, unsigned int max_state, bool prediction);

__global__ 
void calculate_feedback(unsigned int* clauses_feedback, int* scores, unsigned int threshold, float s, unsigned int class_id, bool correct_class, unsigned int clauses_amount, curandState* random_states); 

__global__
void give_feedback_to_clauses(unsigned int* model, unsigned int* clauses_feedback, unsigned int* x_data, bool* clauses_output, unsigned int class_id, unsigned int sample_id, bool correct_class, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state, unsigned int threshold, float s, curandState* random_states);

__global__
void improved_feedback(unsigned int* model, unsigned int* clauses_feedback, unsigned int* x_data, bool* clauses_output, unsigned int class_id, unsigned int sample_id, bool correct_class, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state, unsigned int threshold, float s, curandState* random_states);

__global__
void initialize_random_states(curandState* states, int seed, unsigned int amount_of_states);

