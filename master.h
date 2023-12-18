#define N_LOCATIONS 160
#define TOTAL_PARAM_FLOATS 47576128
#define N_DEVICES 2

typedef struct param_meta {
  int ind;
  int size;
} param_meta;


typedef struct Shared_Struct {
  int n_devices;
  int n_locations;
  int sizes[N_LOCATIONS];
  size_t cum_sizes[N_LOCATIONS];
  size_t total_param_floats;
  bool is_written[N_DEVICES * N_LOCATIONS];
  bool is_reduced[N_LOCATIONS];
  float values[N_DEVICES * TOTAL_PARAM_FLOATS];
} Shared_Struct;


struct thread_data {
    int dev_id;
    int location_ind;
    int param_size;
    float * grad_location_local;
    Shared_Struct * shared_struct;
};

struct reduction_thread_data {
    int location_ind;
    int param_size;
    Shared_Struct * shared_struct;
    unsigned long completed_red_mask;
};