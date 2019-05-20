#include "tsetlin_random_wheel.cuh"
#include <iostream>
#include <stdio.h>
#include <fstream>

class CPUKernel {

public:

  CPUKernel();
  ~CPUKernel();

  // Data loading management
  void load_model(const unsigned int* model, unsigned int classes, unsigned int clauses, unsigned int automatas, unsigned int states);
  void load_training_data(const unsigned int* x_train, const unsigned int* y_train, unsigned int samples);
  void load_validation_data(const unsigned int* x_val, const unsigned int* y_val, unsigned int samples);

  // Debug methods
  void print_model();

  // Training methods
  void fit(int epochs, int batches, bool validation, int threshold, float s, bool feedback = false, bool print_model_after_epoch = false);
  void enable_ssl_s(double delta_s);
  double validate(bool feedback = false);

  private:

  // Training data
  unsigned int* x_train {nullptr};
  unsigned int* y_train {nullptr};
  unsigned int samples_train_n {0};

  // Validation data
  unsigned int* x_val {nullptr};
  unsigned int* y_val {nullptr};
  unsigned int samples_val_n {0};

  // Model related data
  unsigned int* model {nullptr};
  unsigned int classes_amount {0};
  unsigned int clauses_amount {0};
  unsigned int features_amount {0};
  unsigned int automatas_amount {0};
  unsigned int states_amount {0};

  // Training related values
  bool ssl_s_enabled {false};
  double delta_s {0.0};
  
  // Private training methods
  static void train_class_one_epoch(unsigned int class_id, unsigned int batches, unsigned int threshold, float s, unsigned int* model, unsigned int* x_data, unsigned int* y_data, unsigned int samples, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int states_amount, double* training_times, TsetlinRandomWheel* random_generator);
  static void validate_class(unsigned int class_id, unsigned int* model, unsigned int* x_val, unsigned int* y_val, int* scores, unsigned int samples_amount, unsigned int classes_amount, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state);

  static inline void validate_clauses(unsigned int* model, bool* clauses_output, unsigned int* x_data, unsigned int sample_id, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int class_id, unsigned int max_state, bool prediction);
  static inline void reduce_votes(int* scores, unsigned int scores_index, bool* clauses_output, unsigned int clauses_amount, unsigned int threshold);
  static inline void calculate_feedback(unsigned int* clauses_feedback, int* scores, unsigned int threshold, float s, unsigned int class_id, bool correct_class, unsigned int clauses_amount, TsetlinRandomWheel* random_generator);
  static inline void give_feedback_to_clauses(unsigned int* model, unsigned int* clauses_feedback, unsigned int* x_data, bool* clauses_output, unsigned int class_id, unsigned int sample_id, bool correct_class, unsigned int clauses_amount, unsigned int features_amount, unsigned int automatas_amount, unsigned int max_state, unsigned int threshold, float s, TsetlinRandomWheel* random_generator);

  static inline bool automata_action(unsigned int automata_state, unsigned int max_state);
  static inline int get_polarity(int id);
};
