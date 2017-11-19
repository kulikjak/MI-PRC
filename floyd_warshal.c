#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define _CHECK_MATRICES

double _start_in, _start_out;
double _end_in, _end_out;

matrix floyd_warshall_seq(const matrix dm, int32_t size) {
  int32_t i, j, k;

  for (k = 0; k < size; k++)
    for (i = 0; i < size; i++)
      for (j = 0; j < size; j++) {
        dm[i][j] =
            (dm[i][k] + dm[k][j] < dm[i][j]) ? dm[i][k] + dm[k][j] : dm[i][j];
      }

  return dm;
}

matrix floyd_warshall(const matrix dm, int32_t size) {
  int32_t i, j, k;

  #pragma acc data copy(dm[0 : size][0 : size])
  {
    _start_in = omp_get_wtime();  // clock();
    #pragma acc parallel num_gangs(1024) vector_length(128)
    {
      for (k = 0; k < size; k++) {
        #pragma acc loop collapse(2)
        for (i = 0; i < size; i++) {
          for (j = 0; j < size; j++) {
            dm[i][j] = (dm[i][k] + dm[k][j] < dm[i][j]) ? dm[i][k] + dm[k][j]
                                                        : dm[i][j];
          }
        }
      }
    }
    _end_in = omp_get_wtime();  // clock();
  }

  return dm;
}

int main(int argc, char* argv[]) {
  int32_t size;

  FILE* graph_file;
  matrix graph_matrix, distance_matrix, distance_matrix2;

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

  graph_matrix = read_matrix(graph_file, size);
  distance_matrix = get_distance_matrix(graph_matrix, size);

  _start_out = omp_get_wtime();  // clock();
  distance_matrix = floyd_warshall(distance_matrix, size);
  _end_out = omp_get_wtime();  // clock();

#ifdef _CHECK_MATRICES

  printf("Checking, that parallel algorithm runs correctly\n");

  distance_matrix2 = floyd_warshall_seq(distance_matrix, size);
  if (compare_matrices(distance_matrix, distance_matrix2, size)) {
    printf("Both parallel and serial result matrices match.\n");
  } else {
    printf("Serial and parallel result matrices are different.\n");
    printf("Something is wrong!!!.\n");
  }

#endif

  /* print_matrix(distance_matrix, size); */

  free_matrix(graph_matrix, size);
  free_matrix(distance_matrix, size);
  fclose(graph_file);

  printf("== Time: %lf (without data copy)\n", _end_in - _start_in);
  printf("== Time: %lf (with data copy)\n", _end_out - _start_out);

  return EXIT_SUCCESS;
}
