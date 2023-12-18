#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

#include "master.h"


unsigned long set_bit(unsigned long orig, unsigned long bit_location){
  return orig | ((unsigned long) 1 << bit_location);
}

unsigned long clear_bit(unsigned long orig, unsigned long bit_location){
  return orig & ~((unsigned long) 1 << bit_location);
}

bool is_set(unsigned long orig, unsigned long bit_location){
    return (orig >> bit_location) & ((unsigned long) 1);
}


void * do_reduction(void * reduction_thread_inp){

  Shared_Struct * shared_struct = ((struct reduction_thread_data *) reduction_thread_inp) -> shared_struct;
  int n_devices = shared_struct -> n_devices;
  int n_locations = shared_struct -> n_locations;
  int total_param_floats = shared_struct -> total_param_floats;
  size_t * cum_sizes = shared_struct -> cum_sizes;
  
  int location_ind = ((struct reduction_thread_data *) reduction_thread_inp) -> location_ind;
  int param_size = ((struct reduction_thread_data *) reduction_thread_inp) -> param_size;
  
  unsigned long completed_red_mask = ((struct reduction_thread_data *) reduction_thread_inp) -> completed_red_mask;

  bool * is_written = shared_struct -> is_written;
  float * values = shared_struct -> values;

  float * my_reduced_params = (float *) calloc(param_size, sizeof(float));
  
  int d_cnt = 0;
  int loc_in_shared;
  while (d_cnt < n_devices){
    for (int d = 0; d < n_devices; d++){
      if (is_set(completed_red_mask, d)){
        continue;
      }
      if (is_written[d * n_locations + location_ind]){
        loc_in_shared = d * total_param_floats + cum_sizes[location_ind];
        for (int k = 0; k < param_size; k++){
          my_reduced_params[k] += values[loc_in_shared + k];
        }
        d_cnt += 1;
        set_bit(completed_red_mask, d);
      }
    }
  }

  for (int k = 0; k < param_size; k++){
    my_reduced_params[k] = my_reduced_params[k] / n_devices;
  }

  int dev_0_loc = cum_sizes[location_ind];
  memcpy(&values[dev_0_loc], my_reduced_params, param_size * sizeof(float));

  bool * is_reduced = shared_struct -> is_reduced;
  is_reduced[location_ind] = true;

}

void setup_reduction_threads(Shared_Struct * shared_struct){
  int n_devices = shared_struct -> n_devices;
  int n_locations = shared_struct -> n_locations;
  size_t total_param_floats = shared_struct -> total_param_floats;
  int * sizes = shared_struct -> sizes;
  size_t * cum_sizes = shared_struct -> cum_sizes;

  pthread_t threads[n_locations];
  struct reduction_thread_data thread_data_array[n_locations];

  for (int i = 0; i < n_locations; i++){
    thread_data_array[i].location_ind = i;
    thread_data_array[i].param_size = sizes[i];
    thread_data_array[i].shared_struct = shared_struct;
    thread_data_array[i].completed_red_mask = 0;
    pthread_create(&threads[i], NULL, do_reduction, (void *) &thread_data_array[i]);
  }

  for (int i = 0; i < n_locations; i++){
      pthread_join(threads[i], NULL);
  }

}



int main(int argc, char *argv[]) {

  printf("Getting Shared Memory Key\n");
  int shm_id = shmget(IPC_PRIVATE, sizeof(Shared_Struct), IPC_CREAT | 0666);
  if (shm_id < 0) {
    printf("*** shmget error (server) ***\n");
    exit(1);
  }
  printf("Key is: %d\n", shm_id);

  printf("Attaching Shared Memory to Master\n");
  Shared_Struct * shared_struct = (Shared_Struct *) shmat(shm_id, NULL, 0);
  if (shared_struct == (void *) -1) {
      printf("*** shmat error (server) ***\n");
      exit(1);
  }

  // initialize the shared struct from master
  printf("Initializing Shared Memory in Master\n");
  
  shared_struct -> n_devices = N_DEVICES;
  shared_struct -> n_locations = N_LOCATIONS;

  // set parameter sizes (by reading from file which has them strored...)
  FILE * param_size_file = fopen("param_sizes.txt", "r");

  param_meta param_meta_info[N_LOCATIONS];
  fread(&param_meta_info, sizeof(param_meta), N_LOCATIONS, param_size_file);
  fclose(param_size_file);

  for (int i = 0; i < N_LOCATIONS; i++){
    shared_struct -> sizes[param_meta_info[i].ind] = param_meta_info[i].size;
    
  }

  size_t total_size = 0;
  for (int i = 0; i < N_LOCATIONS; i++){
    shared_struct -> cum_sizes[i] = total_size;
    total_size += shared_struct -> sizes[i];
  }
  shared_struct -> total_param_floats = total_size;

  // set that no one has written values yet
  for (int d = 0; d < N_DEVICES; d++){
    for (int i = 0; i < N_LOCATIONS; i++){
      shared_struct -> is_written[d * N_LOCATIONS + i] = false;
    }
  }

  // set that no results are computed yet
  for (int i = 0; i < N_LOCATIONS; i++){
    shared_struct -> is_reduced[i] = false;
  }

  // set all gradient values to 0
  for (int i = 0; i < N_DEVICES * TOTAL_PARAM_FLOATS; i++){
    shared_struct -> values[i] = 0;
  }


  /* FORK THE CHILD EXECUTABLES */

  char * exec_files[] = {"./ResNetCuDNNOpt", "./ResNetMIOpenOpt"};
  int batch_sizes[] = {128, 128};
  char * child_argv[5];
  child_argv[4] = NULL;
  asprintf(&child_argv[0], "%d", N_DEVICES);
  asprintf(&child_argv[2], "%d", shm_id);


  setup_reduction_threads(shared_struct);

  printf("Starting Executables on Devices\n");
  for (int i = 0; i < N_DEVICES; i++){

    int pid = fork();

    if (pid < 0){
      printf("*** fork error ***\n");
      exit(1);
    }

    if (pid == 0){

      asprintf(&child_argv[1], "%d", i);
      asprintf(&child_argv[3], "%d", batch_sizes[i]);

      printf("Starting: %s\n", exec_files[i]);
      execvp(exec_files[i], child_argv);
    }
  }

  free(child_argv[0]);


}