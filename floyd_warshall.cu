#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

//#define _CHECK_MATRICES

#define TILE_SIZE 32
#define LOG_TILE_SIZE 5

clock_t _start_in, _start_out;
clock_t _end_in, _end_out;

// Kernel for independent blocks
__global__ void kernel_independent_blocks(matrix __dm, int32_t __size,
                                          int32_t __b, int32_t k) {
  // get i and j from block and thread information
  const int32_t i = blockIdx.x + (__b << LOG_TILE_SIZE);
  const int32_t j = threadIdx.x + (__b << LOG_TILE_SIZE);

  // check boundaries for matrices with indivisible size
  if (i >= __size || j >= __size) return;

  // calculate values
  __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j]) ? __dm[i][k] + __dm[k][j]
                                                      : __dm[i][j];
}

// Kernel for i-aligned singly depenent blocks
__global__ void kernel_line_dependent_blocks(matrix __dm, int32_t __size,
                                             int32_t __b, int32_t k) {
  // get i, j and ib from block and thread information
  const int32_t ib = blockIdx.x;

  const int32_t i = threadIdx.x + (__b << LOG_TILE_SIZE);
  const int32_t j = threadIdx.y + (ib << LOG_TILE_SIZE);

  // skip already calculated tile
  if (__b == ib) return;

  // check boundaries for matrices with indivisible size
  if (i >= __size || j >= __size) return;

  // calculate values
  __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j]) ? __dm[i][k] + __dm[k][j]
                                                      : __dm[i][j];
}

// Kernel for j-aligned singly depenent blocks
__global__ void kernel_column_dependent_blocks(matrix __dm, int32_t __size,
                                               int32_t __b, int32_t k) {
  // get i, j and jb from block and thread information
  const int32_t jb = blockIdx.x;

  const int32_t i = threadIdx.x + (jb << LOG_TILE_SIZE);
  const int32_t j = threadIdx.y + (__b << LOG_TILE_SIZE);

  // skip already calculated tile
  if (jb == __b) return;

  // check boundaries for matrices with indivisible size
  if (i >= __size || j >= __size) return;

  // calculate values
  __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j]) ? __dm[i][k] + __dm[k][j]
                                                      : __dm[i][j];
}

// Kernel for double depenent blocks
__global__ void kernel_double_dependent_blocks(matrix __dm, int32_t __size,
                                               int32_t __b, int32_t k) {
  // get i, j, ib and jb from block and thread information
  const int32_t ib = blockIdx.y;
  const int32_t jb = blockIdx.x;

  const int32_t i = threadIdx.x + (jb << LOG_TILE_SIZE);
  const int32_t j = threadIdx.y + (ib << LOG_TILE_SIZE);

  // skip already calculated tile
  if (ib == __b || jb == __b) return;

  // check boundaries for matrices with indivisible size
  if (i >= __size || j >= __size) return;

  // calculate values
  __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j]) ? __dm[i][k] + __dm[k][j]
                                                      : __dm[i][j];
}

// Much more clever kernel for double depenent blocks
__global__ void kernel_sm_double_dependent_blocks(matrix __dm, int32_t __size,
                                                  int32_t __b, int32_t k) {
  // TODO
}

void run_algorithm(matrix __dm, int32_t __size) {
  const int32_t s = TILE_SIZE;
  const int32_t tile_count = (__size + s - 1) / s;

  const dim3 tile2D(TILE_SIZE, TILE_SIZE);
  const dim3 grid2D(tile_count, tile_count);

  int32_t k, b;

  for (b = 0; b < tile_count; b++) {
    // Process the independent block first
    for (k = b * s; k < (b + 1) * s; k++) {
      if (k >= __size) break;
      kernel_independent_blocks<<<s, s>>>(__dm, __size, b, k);
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // i-aligned singly depenent blocks
    for (k = b * s; k < (b + 1) * s; k++) {
      if (k >= __size) break;
      kernel_line_dependent_blocks<<<tile_count, tile2D>>>(__dm, __size, b, k);
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // j-aligned singly depenent blocks
    for (k = b * s; k < (b + 1) * s; k++) {
      if (k >= __size) break;
      kernel_column_dependent_blocks<<<tile_count, tile2D>>>(__dm, __size, b,
                                                             k);
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // double dependent blocks
    for (k = b * s; k < (b + 1) * s; k++) {
      if (k >= __size) break;
      kernel_double_dependent_blocks<<<grid2D, tile2D>>>(__dm, __size, b, k);
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
  }
}

int main(int argc, char* argv[]) {
  int32_t size;

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

  if (fscanf(graph_file, "%d", &size) == EOF) {
    printf("Input file is empty.\n");
    return EXIT_FAILURE;
  }

  graph_mtx = read_matrix(graph_file, size);

  hostMtx = get_distance_matrix(graph_mtx, size);

  _start_out = clock();
  devMtx = allocate_and_init_matrix_GPU(hostMtx, size);
  _start_in = clock();

  run_algorithm(devMtx, size);

  _end_in = clock();
  copy_GPU_to_CPU(hostMtx, devMtx, size);
  _end_out = clock();

#ifdef _CHECK_MATRICES

  printf("Checking that parallel algorithm runs correctly\n");

  dist_mtx2 = get_distance_matrix(graph_mtx, size);
  dist_mtx2 = floyd_warshall_seq(dist_mtx2, size);
  if (compare_matrices(hostMtx, dist_mtx2, size)) {
    printf("Both parallel and serial result matrices match.\n");
  } else {
    printf("Serial and parallel result matrices are different.\n");
    printf("Something is wrong!!!.\n");
  }
  free_matrix_CPU(dist_mtx2, size);

#endif

  print_matrix(hostMtx, size);

  free_matrix_GPU(devMtx, size);

  free_matrix_CPU(hostMtx, size);
  free_matrix_CPU(graph_mtx, size);
  fclose(graph_file);

  printf("== Time: %lf (without data copy)\n",
         double(_end_in - _start_in) / CLOCKS_PER_SEC);
  printf("== Time: %lf (with data copy)\n",
         double(_end_out - _start_out) / CLOCKS_PER_SEC);

  return EXIT_SUCCESS;
}
