#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

//#define _CHECK_MATRICES

#define TILE_SIZE 32
#define LOG_TILE_SIZE 5

clock_t _start_in, _start_out;
clock_t _end_in, _end_out;

// Kernel for dijkstra
__global__ void dijsktra( matrix __dm, int __size)
{
  int s = blockIdx.x * blockDim.x + threadIdx.x; // pozice na radku
  if(s >= __size) 
    return;
  
  int32_t i, count, mindistance, nextnode;
  int32_t *visited;
  cudaMalloc(&visited, __size * sizeof(int32_t));
  
  for (i = 0; i < __size; visited[i++] = 0) {}

  visited[s] = 0;

  for (count = 1; count < __size - 1; count++) {
    mindistance = INF;

    for (i = 0; i < __size; i++) {
      if (__dm[s][i] < mindistance && !visited[i]) {
        mindistance = __dm[s][i];
        nextnode = i;
      }
    }

    visited[nextnode] = 1;

    for (i = 0; i < __size; i++) {
      if (!visited[i] && mindistance < INF) {
        if (mindistance + __dm[nextnode][i] < __dm[s][i]) {
          __dm[s][i] = mindistance + __dm[nextnode][i];
        }
      }
    }
  }
  cudaFree(visited);
}

void run_algorithm(matrix , int32_t __size) {
  dijsktra<<<__size / 1024, 1024>>>(matrix, __size);
  cudaDeviceSynchronize(); 
}

int main(int argc, char* argv[]) {
  int32_t __size;

  FILE* graph_file;
  matrix graph_mtx, dist_mtx2;
  matrix hostMtx, devMtx;

  if (argc != 2) {
    printf("Wrong input\n");
    printf("Usage: %s GRAPH_FILE\n", argv[0]);
    return EXIT_FAILURE;
  }

  graph_file = fopen(argv[1], "r");
  if (graph_file == NULL) {
    printf("Cannot open input file.\n");
    return EXIT_FAILURE;
  }

  if (fscanf(graph_file, "%d", &__size) == EOF) {
    printf("Input file is empty.\n");
    return EXIT_FAILURE;
  }

  graph_mtx = read_matrix(graph_file, __size);

  hostMtx = get_distance_matrix(graph_mtx, __size);

  _start_out = clock();
  devMtx = allocate_and_init_matrix_GPU(hostMtx, __size);
  _start_in = clock();

  run_algorithm(devMtx, __size);

  _end_in = clock();
  copy_GPU_to_CPU(hostMtx, devMtx, __size);
  _end_out = clock();

#ifdef _CHECK_MATRICES

  printf("Checking that parallel algorithm runs correctly\n");

  dist_mtx2 = get_distance_matrix(graph_mtx, __size);
  dist_mtx2 = floyd_warshall_seq(dist_mtx2, __size);
  if (compare_matrices(hostMtx, dist_mtx2, __size)) {
    printf("Both parallel and serial result matrices match.\n");
  } else {
    printf("Serial and parallel result matrices are different.\n");
    printf("Something is wrong!!!.\n");
  }
  free_matrix_CPU(dist_mtx2, __size);

#endif

  print_matrix(hostMtx, __size);

  free_matrix_GPU(devMtx, __size);

  free_matrix_CPU(hostMtx, __size);
  free_matrix_CPU(graph_mtx, __size);
  fclose(graph_file);

  printf("== Time: %lf (without data copy)\n",
         double(_end_in - _start_in) / CLOCKS_PER_SEC);
  printf("== Time: %lf (with data copy)\n",
         double(_end_out - _start_out) / CLOCKS_PER_SEC);

  return EXIT_SUCCESS;
}
