#include <cuda.h>

#ifndef TSETLINRANDOMWHEEL_H
#define TSETLINRANDOMWHEEL_H

class TsetlinRandomWheel {

public:

  TsetlinRandomWheel(int seed, unsigned int classes, unsigned int wheel_size);
  ~TsetlinRandomWheel();

  // Public methods
  float get_random_float(unsigned int class_id);

private:

  int seed {0};
  unsigned int classes {0};
  unsigned int wheel_size {0};
  
  float* matrix {nullptr};
  unsigned int* class_indexes {nullptr};

  void create_matrix();

};

#endif
