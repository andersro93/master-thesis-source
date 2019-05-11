
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <thread>
#include "cpu_kernel.cuh"

using namespace std;

CPUKernel::CPUKernel(){

};

void CPUKernel::load_model(const unsigned int* model, unsigned int classes, unsigned int clauses, unsigned int automatas, unsigned int states) {

    // Store the configuration to the kernel class
    this->classes_amount = classes;
    this->clauses_amount = clauses;
    this->automatas_amount = automatas;
    this->features_amount = static_cast<unsigned int>(automatas / 2);
    this->states_amount = states;

    // Check if an model has already been loaded, if so, deallocate its memory
    if(this->model != nullptr) {
        delete [] this->model;
        this->model = nullptr;
    }

    // Allocate memory for the model
    this->model = new unsigned int[classes * clauses * automatas];

    // Copy the data to the kernel memory
    memcpy(this->model, model, sizeof(unsigned int) * classes * clauses * automatas); 
}

void CPUKernel::load_training_data(const unsigned int* x_train, const unsigned int* y_train, unsigned int train_samples_n) {

    // First cleanup if previous training data has been loaded
    if(this->x_train != nullptr) {
        delete [] this->x_train;
        this->x_train = nullptr;
    }
    if(this->y_train != nullptr) {
        delete [] this->y_train;
        this->y_train = nullptr;
    }

    // Allocate memory for training data
    this->x_train = new unsigned int[this->features_amount * train_samples_n];
    this->y_train = new unsigned int[train_samples_n];

    // Copy the data from provided model to kernel
    memcpy(this->x_train, x_train, sizeof(unsigned int) * this->features_amount * train_samples_n);
    memcpy(this->y_train, y_train, sizeof(unsigned int) * train_samples_n);

    // Store how many samples that have been loaded
    this->samples_train_n = train_samples_n;
}

void CPUKernel::load_validation_data(const unsigned int* x_val, const unsigned int* y_val, unsigned int val_samples_n) {

    // First cleanup if previous training data has been loaded
    if(this->x_val != nullptr) {
        delete [] this->x_val;
        this->x_val = nullptr;
    }
    if(this->y_val != nullptr) {
        delete [] this->y_val;
        this->y_val = nullptr;
    }

    // Allocate memory for validation data
    this->x_val = new unsigned int[this->features_amount * val_samples_n];
    this->y_val = new unsigned int[val_samples_n];

    // Copy the data from provided model to kernel
    memcpy(this->x_val, x_val, sizeof(unsigned int) * this->features_amount * val_samples_n);
    memcpy(this->y_val, y_val, sizeof(unsigned int) * val_samples_n);

    // Store how many samples that have been loaded
    this->samples_val_n = val_samples_n;
}

void CPUKernel::fit(int epochs, int batches, bool validation, int threshold, float s, bool feedback, bool print_model_after_epoch)
{

    // Declare an array for the worker threads
    std::thread* worker_threads = new std::thread[this->classes_amount];

    // Start looping the epochs
    for(int epoch = 1; epoch <= epochs; epoch++)
    {
        // Print feedback
        if(feedback == true){
            printf("Starting epoch %d \n", epoch);
        }

        // Start all the worker threads
        for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
            
            // Create a thread for each of the classes that will train and pass the parameters
            worker_threads[class_id] = std::thread(
                    &CPUKernel::train_class_one_epoch, 
                    class_id,
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
                    this->states_amount
                    );
        }

        // Wait for all the threads to finish
        for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
            
            // Create a thread for each of the classes that will train and pass the parameters
            worker_threads[class_id].join();
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
}

double CPUKernel::validate(bool feedback) {

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

    // Create scores array that holds the output from each validation sample
    int* scores = new int[this->classes_amount * this->samples_val_n];

    // Create an array that holds the threads that will validate each of the samples
    std::thread* worker_threads = new std::thread[this->classes_amount];

    // Start the validation for each class
    for(unsigned int class_id = 0; class_id < this->classes_amount; class_id++) {
        
        // Create and start the thread
        worker_threads[class_id] = std::thread(
            &CPUKernel::validate_class,
            class_id,
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
    delete [] scores;

    return accuracy;
}

void CPUKernel::train_class_one_epoch(unsigned int class_id, unsigned int batches, unsigned int threshold, float s, unsigned int* model, unsigned int* x_data, unsigned int* y_data, unsigned int samples, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int states_amount) {

    // Allocate some memory that will be used during training
    // Got some errors using C++ construct syntax, so reverted to C malloc syntax
    bool* clauses_output = (bool*)malloc(sizeof(bool) * clauses_amount);
    int* score = (int*)malloc(sizeof(int));
    unsigned int* clauses_feedback = (unsigned int*)malloc(sizeof(unsigned int) * clauses_amount);

    // Declare some training specific variables
    bool correct_class;

    // Start looping over the batches
    for(unsigned int batch_id = 0; batch_id < batches; batch_id++) {
    
        // Start looping over all of the samples
        for(unsigned int sample_id = 0; sample_id < samples; sample_id++) {
        
            // Check if the current class is the target class for this sample
            correct_class = (class_id == y_data[sample_id]);

            // Check if we are to train the sample on the current class or not
            if(correct_class || ((1.0 * rand())/ 1.0 * RAND_MAX) > (1.0 / classes_amount)) {
                       
                // Evaluate the clause output
                validate_clauses(
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
                reduce_votes(
                        score,
                        0,
                        clauses_output, 
                        class_id, 
                        clauses_amount, 
                        threshold
                );

                // Calculate the feedback to each clause
                calculate_feedback(
                        clauses_feedback,
                        score, 
                        threshold, 
                        s, 
                        class_id, 
                        correct_class, 
                        clauses_amount
                );

                // Perform feedback on the model
                give_feedback_to_clauses(
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
                        s
                );
            }
        }
    }

    // Free up space that was used during training
    free(clauses_output);
    free(score);
    free(clauses_feedback);
}


void CPUKernel::validate_class(unsigned int class_id, unsigned int* model, unsigned int* x_val, unsigned int* y_val, int* scores, unsigned int samples_amount, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state) {

    // Allocate memory for the clause outputs
    bool* clauses_output = (bool*)malloc(sizeof(bool) * clauses_amount);;

    // Start the validation of the samples
    for(unsigned int sample_id = 0; sample_id < samples_amount; sample_id++) {
        
        // Validate the sample
        validate_clauses(
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
        reduce_votes(
                scores,
                ((sample_id * classes_amount) + class_id),
                clauses_output, 
                class_id, 
                clauses_amount, 
                0 
        );

    }

    // Cleanup used memory
    free(clauses_output);
}


void CPUKernel::print_model() {

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

void inline CPUKernel::validate_clauses(unsigned int* model, bool* clauses_output, unsigned int* x_data, unsigned int sample_id, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int class_id, unsigned int max_state, bool prediction) {

    // Declare some variables
    unsigned int sample_value;
    bool action;
    bool all_excluded;
    int automata_polarity;

    // Loop over all of the clauses in the class
    for (unsigned int clause_id = 0; clause_id < clauses_amount; clause_id ++) { 

        // Assume that the clause is true, and then try to prove otherwise
        clauses_output[clause_id] = true;
        all_excluded = true;

        // Loop over each of the automata and "stride" through
        for(unsigned int automata_id = 0; automata_id < automatas_amount; automata_id ++) {

            // Get the action of the automata
            action = automata_action(model[(class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_id], max_state);

            // Check if the automata is in an include state, if so, investigate further...
            if(action == true) {

                // Calculate the polarity of the automata
                automata_polarity = get_polarity(automata_id);

                // Get the sample's value
                sample_value = x_data[(sample_id * features_amount) + (automata_id / 2)]; 

                // Flip the flag that says that all automatas are in exclude mode
                all_excluded = false;

                // Since the automata is in an include state, lets check if the DOES NOT match the desired value
                if(((automata_polarity == 1) && (sample_value != 1)) || ((automata_polarity == -1) && (sample_value != 0))){
                
                    // A condition has been met that would falsify the entire clause. Therefore, evaluate the entire clause to false
                    clauses_output[clause_id] = false;
                    break;
                }
            }
        }

        // If we are in prediction mode, then at least one automata needs to be true, otherwise we will just set it as false
        if(prediction == true && all_excluded == true) {
            clauses_output[clause_id] = false;
        }
    }
}

void inline CPUKernel::reduce_votes(int* scores, unsigned int scores_index, bool* clauses_output, unsigned int class_id, unsigned int clauses_amount, unsigned int threshold) {

    // Store the score tempromary in a variable
    int temp_score = 0;

    for(unsigned int clause_id = 0; clause_id < clauses_amount; clause_id++) {

        temp_score += (get_polarity(clause_id) * clauses_output[clause_id]);
    }

    // Save the score to a given location
    scores[scores_index] = temp_score;
}

void inline CPUKernel::calculate_feedback(unsigned int* clauses_feedback, int* scores, unsigned int threshold, float s, unsigned int class_id, bool correct_class, unsigned int clauses_amount) {
    
    // Declare some private variables
    int clause_polarity;
    int class_score = scores[0];

    // Loop all clauses 
    for (unsigned int clause_id = 0; clause_id < clauses_amount; clause_id++) {
   
        // Determine the polarity of the clause
        clause_polarity = get_polarity(clause_id);

        // Check if we are on the correct class
        if (correct_class == true) {
            
            // Check if we are to skip feedback for this clause
            if(((1.0 * rand() ) / (1.0 * RAND_MAX)) > (((1.0 * threshold) - class_score) / (2.0 * threshold))) {

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
            if(((1.0 * rand() ) / (1.0 * RAND_MAX)) > (((1.0 * threshold) + class_score) / (2.0 * threshold))) {

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
}

void inline CPUKernel::give_feedback_to_clauses(unsigned int* model, unsigned int* clauses_feedback, unsigned int* x_data, bool* clauses_output, unsigned int class_id, unsigned int sample_id, bool correct_class, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state, unsigned int threshold, float s) {

    // Used to calculate the absolute index of an automata
    unsigned int automata_model_index;
    unsigned int automata_temp;

    // Used to tempromary store whether an automata is in include or exclude state 
    bool action;

    // Used to tempromary store the polarity of an automata
    int automata_polarity;

    // Used to tempromary store the feature id of which feature an automata is associated with
    unsigned int sample_value;

    // In case there are more clauses than blocks, we need to loop them 
    for(unsigned int clause_id = 0; clause_id < clauses_amount; clause_id++) {
        
        // Check if we are to do type 1 feedback
        if(clauses_feedback[clause_id] == 1) {
        
            // If the clause output was evaluated to false
            if(clauses_output[clause_id] == 0) {
            
                // Loop and potentially punish all automatas
                for(unsigned int automata_index = 0; automata_index < automatas_amount; automata_index++) {
                
                    // Calculate the position of the current automata
                    automata_model_index = (class_id * clauses_amount * automatas_amount) + (clause_id * automatas_amount) + automata_index;

                    // Get the value for the automata
                    automata_temp = model[automata_model_index];

                    if((automata_temp > 1) && (((1.0 * rand() ) / (1.0 * RAND_MAX)) <= (1.0 / s))) {
                        model[automata_model_index] = automata_temp - 1;
                    }
                }
            }
            else {
            
                // Loop over each of the automatas
                for(unsigned int automata_index = 0; automata_index < automatas_amount; automata_index++){
                
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
                            if((((1.0 * rand() ) / (1.0 * RAND_MAX)) <= ((s - 1.0) / s)) && (automata_temp < max_state)) {
                                model[automata_model_index] = automata_temp + 1;
                            }
                        }
                        // Assumes that the automata is a for automata (since it is not an against automata)
                        else {

                            // Decrement state
                            if((((1.0 * rand() ) / (1.0 * RAND_MAX)) <= (1.0 / s)) && automata_temp > 1) {
                                model[automata_model_index] = automata_temp - 1;
                            }
                        }

                    }
                    // Assumes that the sample is 1 (since it was not 0)
                    else {
                    
                        // Check if the automata is a for automata
                        if(automata_polarity == 1) {
                    
                            // Decrement the state 
                            if((((1.0 * rand() ) / (1.0 * RAND_MAX)) <= ((s - 1.0) / s)) && (automata_temp < max_state)) {
                                model[automata_model_index] = automata_temp + 1;
                            }
                        }
                        // Assumes that the automata is an against automata (since it is not an for automata)
                        else {
                        
                            // Decrement state
                            if((((1.0 * rand() ) / (1.0 * RAND_MAX)) <= (1.0 / s)) && automata_temp > 1) {
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
                for(unsigned int automata_id = 0; automata_id < automatas_amount; automata_id++) {
            
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
}

bool inline CPUKernel::automata_action(unsigned int automata_state, unsigned int max_state) {

    return (automata_state > (static_cast<unsigned int>(max_state / 2))); 
}

int inline CPUKernel::get_polarity(int id){
    
    // Check the id for a "for" polarity
    if ((id & 1) == 0) {

        // Id is a for polarity
        return 1;
    }

    // Previous check failed, therefore the id's polarity must be "against"
    return -1;
}

CPUKernel::~CPUKernel() {


    // Free up memory from the devices
    if(this->model != nullptr) {
        delete [] this->model;
    }

    if(this->x_train != nullptr) {
        delete [] this->x_train;
    }

    if(this->y_train != nullptr) {
        delete [] this->y_train;
    }
    
    if(this->x_val != nullptr) {
        delete [] this->x_val;
    }
    
    if(this->y_val != nullptr) {
        delete [] this->x_val;
    }
}
