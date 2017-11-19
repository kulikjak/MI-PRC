#ifndef __UTILS__
#define __UTILS__

#include <limits.h>
#include <stdbool.h>

#define INF SHRT_MAX

typedef int32_t** matrix;

matrix allocate_matrix(int32_t __size) {
  int32_t i;

  matrix matrix = (int32_t**)malloc(__size * sizeof(int32_t*));
  for (i = 0; i < __size; i++)
    matrix[i] = (int32_t*)malloc(__size * sizeof(int32_t));

  return matrix;
}

matrix get_distance_matrix(const matrix __graph, int32_t __size) {
  int32_t i, j;

  matrix dist = allocate_matrix(__size);

  for (i = 0; i < __size; i++)
    for (j = 0; j < __size; j++)
      dist[i][j] = (i != j && __graph[i][j] == 0) ? INF : __graph[i][j];

  return dist;
}

matrix read_matrix(FILE* __file, int32_t __size) {
  int32_t value;
  int32_t i, j;

  matrix mtx = allocate_matrix(__size);

  for (i = 0; i < __size; i++)
  for (j = 0; j < __size; j++) {
    do {
      value = getc(__file);
    } while (value == (unsigned char)' ' || value == (unsigned char)'\t' ||
             value == (unsigned char)'\n');
    mtx[i][j] = value - '0';
  }
  return mtx;
}

void print_matrix(const matrix __mtx, int32_t __size) {
  int32_t i, j;

  for (i = 0; i < __size; i++) {
    for (j = 0; j < __size; j++) {
      if (__mtx[i][j] == INF)
        printf("%4s", "INF");
      else
        printf("%4d", __mtx[i][j]);
    }
    printf("\n");
  }
}

bool compare_matrices(const matrix __a, const matrix __b, int32_t __size) {
  int32_t i, j;

  for (i = 0; i < __size; i++)
    for (j = 0; j < __size; j++)
      if (__a[i][j] != __b[i][j]) return false;
  return true;
}

void free_matrix(matrix __mtx, int32_t __size) {
  int32_t i;

  for (i = 0; i < __size; i++)
    free(__mtx[i]);
  free(__mtx);
}

#endif
