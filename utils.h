#ifndef __UTILS__
#define __UTILS__

#include <limits.h>

#define INF SHRT_MAX

typedef int8_t** matrix;

matrix allocate_matrix(int32_t size) {
  int32_t i;

  matrix matrix = (int8_t**)malloc(size * sizeof(int8_t*));
  for (i = 0; i < size; i++)
    matrix[i] = (int8_t*)malloc(size * sizeof(int8_t));

  return matrix;
}

matrix get_distance_matrix(const matrix graph_matrix, int32_t size) {
  int32_t i, j;
  matrix distance_matrix = allocate_matrix(size);

  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++)
      distance_matrix[i][j] =
          (i != j && graph_matrix[i][j] == 0) ? INF : graph_matrix[i][j];

  return distance_matrix;
}

matrix read_matrix(FILE* graph_file, int32_t size) {
  int8_t value;
  int32_t i, j;

  matrix graph_matrix = allocate_matrix(size);

  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++) {
      do {
        value = getc(graph_file);
      } while (value == (unsigned char)' ' || value == (unsigned char)'\t' ||
               value == (unsigned char)'\n');
      graph_matrix[i][j] = value - '0';
    }
  return graph_matrix;
}

void print_matrix(const matrix distance_matrix, int32_t size) {
  int32_t i, j;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (distance_matrix[i][j] == INF)
        printf("%4s", "INF");
      else
        printf("%4d", distance_matrix[i][j]);
    }
    printf("\n");
  }
}

void free_matrix(matrix matrix, int32_t size) {
  int32_t i;

  for (i = 0; i < size; i++) free(matrix[i]);
  free(matrix);
}

#endif
