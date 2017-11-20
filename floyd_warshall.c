#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define _CHECK_MATRICES

double _start_in, _start_out;
double _end_in, _end_out;


matrix floyd_warshall(const matrix __dm, int32_t __size) {
  int32_t i, j, k;

  #pragma acc data copy(__dm[0 : __size][0 : __size])
  {
    _start_in = omp_get_wtime();  // clock();
    #pragma acc parallel num_gangs(1024) vector_length(128)
    {
      for (k = 0; k < __size; k++) {
        #pragma acc loop collapse(2)
        for (i = 0; i < __size; i++) {
          for (j = 0; j < __size; j++) {
            __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                             ? __dm[i][k] + __dm[k][j]
                             : __dm[i][j];
          }
        }
      }
    }
    _end_in = omp_get_wtime();  // clock();
  }

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

  _start_out = omp_get_wtime();  // clock();
  dist_mtx = floyd_warshall(dist_mtx, size);
  _end_out = omp_get_wtime();  // clock();

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
