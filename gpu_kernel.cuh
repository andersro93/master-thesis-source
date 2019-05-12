#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include "kernels.cuh"
#include "tsetlin_random_wheel.cuh"

class GPUKernel {

public:

  GPUKernel();
  ~GPUKernel();

  // GPU Management
  void enable_gpu(unsigned int gpu_id);
  void remove_gpu(unsigned int gpu_id);

  // Data loading management
  void load_model(const unsigned int* model, unsigned int classes, unsigned int clauses, unsigned int automatas, unsigned int states);
  void load_training_data(const unsigned int* x_train, const unsigned int* y_train, unsigned int samples);
  void load_validation_data(const unsigned int* x_val, const unsigned int* y_val, unsigned int samples);

  // Debug methods
  void print_model();

  // Training methods
  void fit(int epochs, int batches, bool validation, int threshold, float s, bool feedback = false, bool print_model_after_epoch = false);
  double validate(bool feedback = false);

  private:

  // GPU configuration
  std::vector<unsigned int> enabled_gpus;

  // Training data
  unsigned int* x_train {nullptr};
  unsigned int* y_train {nullptr};
  unsigned int samples_train_n {0};

  // Validation data
  unsigned int* x_val {nullptr};
  unsigned int* y_val {nullptr};
  unsigned int samples_val_n {0};

  // Random generator related data
  bool random_states_initialized {false};
  curandState* random_states {nullptr};

  // Model related data
  unsigned int* model {nullptr};
  unsigned int classes_amount {0};
  unsigned int clauses_amount {0};
  unsigned int features_amount {0};
  unsigned int automatas_amount {0};
  unsigned int states_amount {0};

  // Private training methods
  static void train_class_one_epoch(unsigned int class_id, unsigned int gpu_id, unsigned int batches, unsigned int threshold, float s, unsigned int* model, unsigned int* x_data, unsigned int* y_data, unsigned int samples, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int states_amount, double* training_times, TsetlinRandomWheel* random_generator);
  static void validate_class(unsigned int class_id, unsigned int gpu_id, unsigned int* model, unsigned int* x_val, unsigned int* y_val, int* scores, unsigned int samples_amount, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state);

  // GPU related methods
  void select_gpu(unsigned int gpu_id);
  static dim3 calculate_blocks_per_kernel(unsigned int clauses_amount);
  static dim3 calculate_threads_per_block(unsigned int automatas_amount);

};
