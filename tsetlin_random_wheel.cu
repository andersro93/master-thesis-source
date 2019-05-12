#include <stdlib.h>
#include "tsetlin_random_wheel.cuh"

TsetlinRandomWheel::TsetlinRandomWheel(int seed, unsigned int classes, unsigned int wheel_size)
{
    // Store the values to the object
    this->seed = seed;
    this->classes = classes;
    this->wheel_size = wheel_size;

    // Initialize the matrix wheel
    this->create_matrix();
}

float TsetlinRandomWheel::get_random_float(unsigned int class_id) {

    // Increment the index of the class
    this->class_indexes[class_id] += 1;

    // Return the element at the calculated index
    return this->matrix[((class_id * this->wheel_size) + ((this->class_indexes[class_id]-1) % this->wheel_size))];
}

void TsetlinRandomWheel::create_matrix() {

    // Initialize the matrix
    this->matrix = (float*) malloc(sizeof(float) * this->classes * this->wheel_size);

    // Generate random float values in all the fields
    for(unsigned int index = 0; index < this->classes * this->wheel_size; index++) {

        this->matrix[index] = (float)rand()/RAND_MAX;
    }

    // Initialize the indexes
    this->class_indexes = (unsigned int*) malloc(sizeof(unsigned int) * this->classes);

    for(unsigned int index = 0; index < this->classes; index++) {

        this->class_indexes[index] = 0;
    }
}

TsetlinRandomWheel::~TsetlinRandomWheel()
{
    free(this->matrix);
    free(this->class_indexes);
}