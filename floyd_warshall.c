#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define _CHECK_MATRICES

double _start_in, _start_out;
double _end_in, _end_out;

matrix floyd_warshall(const matrix __dm, int32_t __n) {
  int32_t i, j, k, b;

  #pragma acc data copy(__dm[0 : __n][0 : __n])
  {
    _start_in = omp_get_wtime();

    #pragma acc parallel
    {
      // i-aligned singly depenent blocks
      #pragma acc loop independent collapse(2)
      for (b = 0; b < __n; b++)
      for (j = 0; j < __n; j++) {
        __dm[b][j] = (__dm[b][b] + __dm[b][j] < __dm[b][j])
                         ? __dm[b][b] + __dm[b][j]
                         : __dm[b][j];
      }

      // j-aligned singly depenent blocks
      #pragma acc loop independent collapse(2)
      for (b = 0; b < __n; b++)
      for (i = 0; i < __n; i++) {
        __dm[i][b] = (__dm[i][b] + __dm[b][b] < __dm[i][b])
                         ? __dm[i][b] + __dm[b][b]
                         : __dm[i][b];
      }
    }

    // double dependent blocks
    for (k = 0; k < __n; k++) {
      #pragma acc parallel loop collapse(2) num_gangs(32) vector_length(1024)
      for (i = 0; i < __n; i++)
      for (j = 0; j < __n; j++) {
        __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                         ? __dm[i][k] + __dm[k][j]
                         : __dm[i][j];
      }
    }

    _end_in = omp_get_wtime();
  }
  return __dm;
}

matrix floyd_warshall_blocked(const matrix __dm, int32_t __n, int32_t __s) {
  int32_t i, j, k, b, ib, jb;

  _start_in = omp_get_wtime();

  for (b = 0; b < __n / __s; b++) {

    // Process the independent block first
    for (k = b * __s; k < (b + 1) * __s; k++)
    for (i = b * __s; i < (b + 1) * __s; i++)
        for (j = b * __s; j < (b + 1) * __s; j++) {
          __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                           ? __dm[i][k] + __dm[k][j]
                           : __dm[i][j];
        }

    // i-aligned singly depenent blocks
    for (ib = 0; ib < __n / __s; ib++) {
      if (ib == b) continue;
      for (k = b * __s; k < (b + 1) * __s; k++)
      for (i = b * __s; i < (b + 1) * __s; i++)
      for (j = ib * __s; j < (ib + 1) * __s; j++) {
        __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                         ? __dm[i][k] + __dm[k][j]
                         : __dm[i][j];
      }
    }

    // j-aligned singly depenent blocks
    for (jb = 0; jb < __n / __s; jb++) {
      if (jb == b) continue;
      for (k = b * __s; k < (b + 1) * __s; k++)
      for (i = jb * __s; i < (jb + 1) * __s; i++)
      for (j = b * __s; j < (b + 1) * __s; j++) {
        __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                         ? __dm[i][k] + __dm[k][j]
                         : __dm[i][j];
      }
    }

    // double dependent blocks
    for (ib = 0; ib < __n / __s; ib++)
    for (jb = 0; jb < __n / __s; jb++) {
      if (ib == b || jb == b) continue;
      for (i = jb * __s; i < (jb + 1) * __s; i++)
      for (j = ib * __s; j < (ib + 1) * __s; j++)
      for (k = b * __s; k < (b + 1) * __s; k++) {
        __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                         ? __dm[i][k] + __dm[k][j]
                         : __dm[i][j];
      }
    }
  }
  _end_in = omp_get_wtime();

  return __dm;
}

int main(int argc, char* argv[]) {
  int32_t size;

  FILE* graph_file;
  matrix graph_mtx, dist_mtx, dist_mtx2;

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
  dist_mtx = get_distance_matrix(graph_mtx, size);

  _start_out = omp_get_wtime();
  dist_mtx = floyd_warshall(dist_mtx, size);
  _end_out = omp_get_wtime();

#ifdef _CHECK_MATRICES

  printf("Checking that parallel algorithm runs correctly\n");

  dist_mtx2 = get_distance_matrix(graph_mtx, size);
  dist_mtx2 = floyd_warshall_seq(dist_mtx2, size);
  if (compare_matrices(dist_mtx, dist_mtx2, size)) {
    printf("Both parallel and serial result matrices match.\n");
  } else {
    printf("Serial and parallel result matrices are different.\n");
    printf("Something is wrong!!!.\n");
  }
  free_matrix(dist_mtx2, size);

#endif

  /* print_matrix(dist_mtx, size); */

  free_matrix(graph_mtx, size);
  free_matrix(dist_mtx, size);
  fclose(graph_file);

  printf("== Time: %lf (without data copy)\n", _end_in - _start_in);
  printf("== Time: %lf (with data copy)\n", _end_out - _start_out);

  return EXIT_SUCCESS;
}
