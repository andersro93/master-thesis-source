#include "main.cuh"
#include <iostream>
#include <string>

// Some constants that will be used throughout the experiment
#define RANDOM_SEED 1337
#define RANDOM_WHEEL 10000
#define EPOCHS 1000
#define DELTA_S 1.0
#define VALIDATION true
#define DEBUG_MODE true
#define PRINT_MODEL false

// Dataset IDs
#define DATASET_NOISYXOR 1
#define DATASET_IMDB 2
#define DATASET_IRIS 3
#define DATASET_MNIST 4

// Compute methods
#define COMPUTE_CPU 1
#define COMPUTE_GPU 2

int main(int argc, const char* argv[])
{
	// Start with a splash
    print_splash();

    // Validate the arguments
    if(validate_arguments(argc, argv) == false) {
        exit(0);
    }

    // Load the given dataset
    dataset data = load_dataset(std::stoi(argv[2]));

    // Create an initial model
    model model = get_model(std::stoi(argv[2]));

    printf("===============================================================\n\n");
    printf("Configuration: \n");

    // Set a random seed that will be used for random number generation
    std::srand(RANDOM_SEED);
    printf("\tRandom seed: %d \n", RANDOM_SEED);

    if(std::stoi(argv[2]) == DATASET_NOISYXOR) {
        printf("\tDataset: Noisy Xor \n");
    }
    else if(std::stoi(argv[2]) == DATASET_IMDB){
        printf("\tDataset: IMDB \n");
    }
    else if(std::stoi(argv[2]) == DATASET_IRIS){
        printf("\tDataset: IRIS \n");
    }
    else if(std::stoi(argv[2]) == DATASET_MNIST){
        printf("\tDataset: MNIST \n");
    }
    else {
        printf("\tDataset: Unknown \n");
    }

    // Print information about the dataset
    printf("\tEpochs %d \n", EPOCHS);
    printf("\tBatches %d \n", data.batches);
    printf("\tSamples Training: %d \n", data.samples_training);
    printf("\tSamples Validation: %d \n", data.samples_validation);

    // Print information about the model
    printf("\nModel configuration: \n");
    printf("\tClasses: %d \n", model.classes);
    printf("\tClauses: %d \n", model.clauses);
    printf("\tFeatures: %d \n", model.features);
    printf("\tAutomata: %d \n", model.automatas);
    printf("\tMax State: %d \n", model.max_states);

    // Check if we are running in CPU mode
    if(std::stoi(argv[1]) == COMPUTE_CPU) {

        // Print info
        printf("\nCompute Method: CPU \n\n");
        printf("===============================================================\n\n");

        // Create a CPU kernel
        CPUKernel kernel = CPUKernel();

        // Load the created model
        kernel.load_model(model.data, model.classes, model.clauses, model.automatas, model.max_states);

        // Load training and validation data
        kernel.load_training_data(data.x_data_train, data.y_data_train, data.samples_training);
        kernel.load_validation_data(data.x_data_validation, data.y_data_validation, data.samples_validation);

        // Check if we are using experimental S
        if(std::stoi(argv[3]) == 2) {
            
            printf("Experimental S enabled with an delta of %f \n", DELTA_S);
            kernel.enable_ssl_s(DELTA_S);
        }

        // Start the fitting of the model
        kernel.fit(EPOCHS, data.batches, VALIDATION, data.threshold, data.s, DEBUG_MODE, PRINT_MODEL);
    }
    else if(std::stoi(argv[1]) == COMPUTE_GPU) {

        // Print info
        printf("\nCompute Method: GPU \n\n");
        printf("===============================================================\n\n");

        // Create a GPU kernel
        GPUKernel kernel = GPUKernel();

        // Lets enable the first n GPUs
        for(unsigned int gpu_id = 0; gpu_id < std::stoi(argv[4]); gpu_id++) {

            // Enable the GPU in the kernel
            kernel.enable_gpu(gpu_id);
        }

        // Check if we are using experimental S
        if(std::stoi(argv[3]) == 2) {
            
            printf("Experimental S enabled with an delta of %f \n", DELTA_S);
            kernel.enable_ssl_s(DELTA_S);
        }

        // Load the created model
        kernel.load_model(model.data, model.classes, model.clauses, model.automatas, model.max_states);

        // Load training and validation data
        kernel.load_training_data(data.x_data_train, data.y_data_train, data.samples_training);
        kernel.load_validation_data(data.x_data_validation, data.y_data_validation, data.samples_validation);

        // Start the fitting of the model
        kernel.fit(EPOCHS, data.batches, VALIDATION, data.threshold, data.s, DEBUG_MODE, PRINT_MODEL);

    }

    // Reset cuda device
    cudaDeviceReset();

    return 0;
}

dataset load_dataset(int dataset_id) {

    if(dataset_id == DATASET_NOISYXOR) {
        return load_noisyxor_dataset();
    }
    else if(dataset_id == DATASET_IMDB) {
        return load_imdb_dataset();
    }
    else if(dataset_id == DATASET_IRIS) {
        return load_iris_dataset();
    }
    else if(dataset_id == DATASET_MNIST) {
        return load_mnist_dataset();
    }

    // Invalid dataset must have been provided, just exit
    exit(1);
}

model get_model(int dataset_id) {

    if(dataset_id == DATASET_NOISYXOR) {
        // 2 classes, 10 Clauses, 12 features, 24 automata, 100 states
        return create_model(2, 10, 12, 24, 100);
    }
    else if(dataset_id == DATASET_IMDB) {
        // 2 classes, 2000 Clauses, 5000 features, 10000 automata, 500 states
        return create_model(2, 2000, 5000, 10000, 500);
    }
    else if(dataset_id == DATASET_IRIS) {
        // 3 classes, 300 Clauses, 16 features, 32 automata, 100 states
        return create_model(3, 300, 16, 32, 100);
    }
    else if(dataset_id == DATASET_MNIST) {
        // 10 classes, 2000 clauses, 784 features, 1568 automata, 100 states
        return create_model(10, 2000, 784, 1568, 100);
    }

    // Invalid dataset must have been provided, just exit
    exit(1);
}

dataset load_imdb_dataset() 
{
    // Create the dataset struct
    dataset data = {
        5000,           // Amount of features on the samples
        25000,          // Amount of training samples
        25000,          // Amount of validation samples
        1,              // Amount of batches per epoch
        27.0,           // Initial S value
        40              // Threshold value
    };

    // Load the training data from file
    load_from_file(&data, std::string("./datasets/IMDBTrainingData.txt"), std::string("./datasets/IMDBTestData.txt"));

    return data;
}

dataset load_mnist_dataset() {

    // Create the dataset struct
    dataset data = {
        784,            // Amount of features on the samples
        60000,          // Amount of training samples
        10000,          // Amount of validation samples
        1,              // Amount of batches per epoch
        10.0,           // Initial S value
        50              // Threshold value
    };

    // Load the training data from file
    load_from_file(&data, std::string("./datasets/MNISTTraining.txt"), std::string("./datasets/MNISTTest.txt"));

    return data;
}

dataset load_iris_dataset()
{
    // Create the dataset struct
    dataset data = {
        16,             // Amount of features on the samples
        120,            // Amount of training samples
        30,             // Amount of validation samples
        100,            // Amount of batches per epoch
        3.0,            // Initial S value
        10              // Threshold value
    };

    // Load the training data from file
    load_from_file(&data, std::string("./datasets/BinaryIrisTrainingData.txt"), std::string("./datasets/BinaryIrisTestData.txt"));

    return data;
}

dataset load_noisyxor_dataset()
{
    // Create the dataset struct
    dataset data = {
        12,             // Amount of features on the samples
        5000,           // Amount of training samples
        5000,           // Amount of validation samples
        200,            // Amount of batches per epoch
        3.9,            // Initial S value
        25              // Threshold value
    };

    // Load the training data from file
    load_from_file(&data, std::string("./datasets/NoisyXORTrainingData.txt"), std::string("./datasets/NoisyXORTestData.txt"));

    return data;
}

void load_from_file(dataset* data, std::string training_file, std::string validation_file) {

        // Some file reading related pointers
    FILE* fp {nullptr};
    char* line {nullptr};
    size_t position {0};
    const char* seperator = " ";
	char* token {nullptr};

    // Open a file reader for the training data
    fp = fopen(training_file.c_str(), "r");

    // Check if we were successfull in reading the file contents
	if (fp == nullptr) {
		printf("Unable to open the training data. Please check that the file is correctly placed in the datasets folder.\n");
		exit(EXIT_FAILURE);
	}

    // Initialize the arrays that will contain the training data
    data->x_data_train = (unsigned int*) malloc(sizeof(unsigned int*) * data->samples_training * data->features);
    data->y_data_train = (unsigned int*) malloc(sizeof(unsigned int) * data->samples_training);

    // Read inn all the training data
	for (int sample_index = 0; sample_index < data->samples_training; sample_index++) {
		getline(&line, &position, fp);

		token = strtok(line, seperator);
		for (int feature_index = 0; feature_index < data->features; feature_index++) {
			data->x_data_train[(sample_index * data->features) + feature_index] = std::stoi(token);
			token=strtok(NULL, seperator);
		}
		data->y_data_train[sample_index] = std::stoi(token);
	}

    // Open a file reader for the validation data
    // Reset and initialize some pointers to zero
    fp = fopen(validation_file.c_str(), "r");
	line = nullptr;
	position = 0;
	token = nullptr;

    // Check if we were successfull in reading the file contents
	if (fp == nullptr) {
		printf("Unable to open the testing data. Please check that the file is correctly placed in the datasets folder.\n");
		exit(EXIT_FAILURE);
	}

    // Initialize the arrays that will contain the training data
    data->x_data_validation = (unsigned int*) malloc(sizeof(unsigned int*) * data->features * data->samples_validation);
    data->y_data_validation = (unsigned int*) malloc(sizeof(unsigned int*) * data->samples_validation);

    // Read inn all the training data
	for (int sample_index = 0; sample_index < data->samples_validation; sample_index++) {
		getline(&line, &position, fp);

		token = strtok(line, seperator);
		for (int feature_index = 0; feature_index < data->features; feature_index++) {
			data->x_data_validation[(sample_index * data->features) + feature_index] = std::stoi(token);
			token=strtok(NULL, seperator);
		}
		data->y_data_validation[sample_index] = std::stoi(token);
	}
}

model create_model(unsigned int classes, unsigned int clauses, unsigned int features, unsigned int automatas, unsigned int max_states)
{
    // Create the model object
    model model = {
        classes,
        clauses,
        features,
        automatas,
        max_states
    };

    // Allocate storage for the model
    model.data = (unsigned int*) malloc(sizeof(unsigned int) * classes * clauses * automatas);

    // Randomize the states of the model
    for(unsigned int index = 0; index < (classes * clauses * automatas); index += 1)
    {
        // Draw a random number between the max state and 1
        model.data[index] = rand() % max_states + 1;
    }

    return model;
}

bool validate_arguments(int argc, const char* argv[]) {

    if(argc < 4) {

        print_instructions();
        return false;
    }

    if(std::stoi(argv[1]) > 2 || std::stoi(argv[1]) < 1) {
        printf("The first argument must either be 1 or 2 \n");
        return false;
    }

    if(std::stoi(argv[2]) > 4 || std::stoi(argv[2]) < 1) {
        printf("The second argument must be a valid dataset. See instructions for more information \n");
        return false;
    }

    if(std::stoi(argv[3]) > 2 && std::stoi(argv[3]) < 1) {
        printf("The third argument must either be 1 (off) or 2 (on) if you want to use self adjusting s or not \n");
        return false;
    }

    if((std::stoi(argv[1]) == 2) && (std::stoi(argv[4]) > 2 && std::stoi(argv[4]) < 1)) {
        printf("The fourth argument must be a number larger than 0 and less than the amount of available GPUs on the system \n");
        return false;
    }


    return true;
}

void print_instructions()
{
    printf("Instructions on how to use this code: \n\n");
    printf("This code accepts arguments to manipulate which configuration to run.\n\n");
    printf("First argument (compute device): \n");
    printf("\t- %d (Run with CPU)\n", COMPUTE_CPU);
    printf("\t- %d (Run with GPU)\n", COMPUTE_GPU);

    printf("\nSecond argument (dataset and configuration): \n");
    printf("\t- %d (Noisy xor)\n", DATASET_NOISYXOR);
    printf("\t- %d (IMDB)\n", DATASET_IMDB);
    printf("\t- %d (Binary iris)\n", DATASET_IRIS);
    printf("\t- %d (MNIST) \n", DATASET_MNIST);

    printf("\nThird argument (If to enable self adjusting S or not): \n");
    printf("\t - 1 (off) \n");
    printf("\t - 2 (on) \n");

    printf("\nForth argument (how many GPUs to enable, if GPU is selected): \n");
    printf("\t - Any number larger than 0\n");
}

void print_splash()
{
    printf("\n");
    printf(R"EOF( _____  __  ___ _____ _   _ __  _   __ __  __   ____  _ _ __  _ ___        ___  __   ___ 
|_   _/' _/| __|_   _| | | |  \| | |  V  |/  \ / _/ || | |  \| | __|  __  | _,\/__\ / _/ 
  | | `._`.| _|  | | | |_| | | ' | | \_/ | /\ | \_| >< | | | ' | _|  |__| | v_/ \/ | \__ 
  |_| |___/|___| |_| |___|_|_|\__| |_| |_|_||_|\__/_||_|_|_|\__|___|      |_|  \__/ \__/ 
  
  )EOF");
    printf("Copyright (c) 2019 Anders Refsdal Olsen \n\n");
    printf("Permission is hereby granted, free of charge, to any person obtaining a copy \n");
    printf("of this software and associated documentation files (the \"Software\"), to deal \n");
    printf("in the Software without restriction, including without limitation the rights \n");
    printf("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell \n");
    printf("copies of the Software, and to permit persons to whom the Software is \n");
    printf("furnished to do so, subject to the following conditions: \n\n");
    printf("\tThe above copyright notice and this permission notice shall be included in all \n");
    printf("\tcopies or substantial portions of the Software. \n\n");

    printf("Source of this code is available at: https://github.com/andersro93/master-thesis-source \n\n");
}
