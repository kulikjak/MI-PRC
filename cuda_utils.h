#ifndef __UTILS__
#define __UTILS__

#include <limits.h>
#include <stdbool.h>
#include <string.h>

#define INF SHRT_MAX

typedef int32_t** matrix;

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

matrix allocate_matrix_CPU(int32_t __size) {
  int32_t i;

  matrix mtx = (int32_t**)malloc(__size * sizeof(int32_t*));
  for (i = 0; i < __size; i++)
    mtx[i] = (int32_t*)malloc(__size * sizeof(int32_t));

  return mtx;
}

matrix allocate_and_init_matrix_GPU(const matrix __hostMtx, int32_t __size) {
  int32_t i;
  int32_t* ptr;
  matrix devMtx;

  HANDLE_ERROR(cudaMalloc(&devMtx, __size * sizeof(*devMtx)));

  for (i = 0; i < __size; i++) {
    HANDLE_ERROR(cudaMalloc(&ptr, __size * sizeof(*ptr)));

    // copy data from CPU to GPU if matrix is given
    if (__hostMtx)
      HANDLE_ERROR(cudaMemcpy(ptr, __hostMtx[i], __size * sizeof(*ptr),
                              cudaMemcpyHostToDevice));

    // copy pointer to this row to GPU matrix
    HANDLE_ERROR(
        cudaMemcpy(&devMtx[i], &ptr, sizeof(ptr), cudaMemcpyHostToDevice));
  }

  return devMtx;
}

void copy_CPU_to_GPU(const matrix __hostMtx, matrix __devMtx, int32_t __size) {
  // allocate aux array for GPU line pointers
  int32_t i;
  int32_t** dev_line = (int32_t**)malloc(__size * sizeof(int32_t*));

  // copy all GPU line pointers to CPU
  HANDLE_ERROR(cudaMemcpy(dev_line, __devMtx, __size * sizeof(*__devMtx),
                          cudaMemcpyDeviceToHost));
  for (i = 0; i < __size; i++)
    // copy data to GPU
    HANDLE_ERROR(cudaMemcpy(dev_line[i], __hostMtx[i],
                            __size * sizeof(*dev_line[i]),
                            cudaMemcpyHostToDevice));

  free(dev_line);
}

void copy_GPU_to_CPU(matrix __hostMtx, const matrix __devMtx, int32_t __size) {
  // allocate aux array for GPU line pointers
  int32_t i;
  int32_t** dev_line = (int32_t**)malloc(__size * sizeof(int32_t*));

  // copy all GPU line pointers to CPU
  HANDLE_ERROR(cudaMemcpy(dev_line, __devMtx, __size * sizeof(*__devMtx),
                          cudaMemcpyDeviceToHost));
  for (i = 0; i < __size; i++)
    // copy data to CPU
    HANDLE_ERROR(cudaMemcpy(__hostMtx[i], dev_line[i],
                            __size * sizeof(*dev_line[i]),
                            cudaMemcpyDeviceToHost));

  free(dev_line);
}

void copy_matrix(matrix __desc, const matrix __src, int32_t __size) {
  int32_t i;

  for (i = 0; i < __size; i++) memcpy(__desc[i], __src[i], __size);
}

matrix get_distance_matrix(const matrix __graph, int32_t __size) {
  int32_t i, j;

  matrix dist = allocate_matrix_CPU(__size);

  for (i = 0; i < __size; i++)
    for (j = 0; j < __size; j++)
      dist[i][j] = (i != j && __graph[i][j] == 0) ? INF : __graph[i][j];

  return dist;
}

matrix read_matrix(FILE* __file, int32_t __size) {
  int32_t value;
  int32_t i, j;

  matrix mtx = allocate_matrix_CPU(__size);

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
    for (j = 0; j < __size; j++) {
      if (__a[i][j] != __b[i][j]) {
        printf("%d %d %d %d\n", __a[i][j], __b[i][j], i, j);
        return false;
      }
    }
  return true;
}

matrix floyd_warshall_seq(const matrix __dm, int32_t __size) {
  int32_t i, j, k;

  for (k = 0; k < __size; k++)
    for (i = 0; i < __size; i++)
      for (j = 0; j < __size; j++) {
        __dm[i][j] = (__dm[i][k] + __dm[k][j] < __dm[i][j])
                         ? __dm[i][k] + __dm[k][j]
                         : __dm[i][j];
      }
  return __dm;
}

void free_matrix_CPU(matrix __mtx, int32_t __size) {
  int32_t i;

  for (i = 0; i < __size; i++) free(__mtx[i]);
  free(__mtx);
}

void free_matrix_GPU(matrix __mtx, int32_t __size) {
  int32_t i;
  int32_t* ptr;

  for (i = 0; i < __size; i++) {
    // copy devPtr to host pc and free it (cannot be done directly)
    HANDLE_ERROR(
        cudaMemcpy(&ptr, &__mtx[i], sizeof(ptr), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptr));
  }
  HANDLE_ERROR(cudaFree(__mtx));
}

#endif
