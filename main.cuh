#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "cpu_kernel.cuh"
#include "gpu_kernel.cuh"

struct dataset {

    // The data related to the dataset
    unsigned int* x_data_train {nullptr};
    unsigned int* y_data_train {nullptr};
    unsigned int* x_data_validation {nullptr};
    unsigned int* y_data_validation {nullptr};
    unsigned int features {0};
    unsigned int samples_training {0};
    unsigned int samples_validation {0};

    // Just have some initial values for the dataset
    unsigned int threshold {0};
    double s {0.0};

    dataset(unsigned int features, unsigned int samples_training, unsigned int samples_validation, double s, unsigned int threshold){
        this->features = features;
        this->samples_training = samples_training;
        this->samples_validation = samples_validation;
        this->s = s;
        this->threshold = threshold;
    }

    ~dataset(){
        if(x_data_train){
            delete [] x_data_train;
        }
        if(y_data_train){
            delete [] y_data_train;
        }
        if(x_data_validation){
            delete [] x_data_validation;
        }
        if(y_data_validation){
            delete [] y_data_validation;
        }
    }
};

struct model {
    unsigned int* data {nullptr};
    unsigned int classes {0};
    unsigned int clauses {0};
    unsigned int features {0};
    unsigned int automatas {0};
    unsigned int max_states {0};

    model(unsigned int classes, unsigned int clauses, unsigned int features, unsigned int automatas, unsigned int max_states){
        this->classes = classes;
        this->clauses = clauses;
        this->features = features;
        this->automatas = automatas;
        this->max_states = max_states;
    }

    ~model(){
        if(data){
            delete [] data;
        }
    }
};

// Main method
int main( int argc, const char* argv[] );

// Model helper methods
model get_model(int dataset_id);
model create_model(unsigned int classes, unsigned int clauses, unsigned int features, unsigned int automatas, unsigned int max_states);

// Helper methods for loading and deleting datasets
dataset load_dataset(int dataset_id);
dataset load_imdb_dataset() ;
dataset load_noisyxor_dataset();
dataset load_iris_dataset();
dataset load_mnist_dataset();

void load_from_file(dataset* data, std::string training_file, std::string validation_file);

// Helper / debug methods for printing info
bool validate_arguments(int argc, const char* argv[]);
void print_instructions();
void print_splash();
