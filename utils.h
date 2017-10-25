#ifndef __UTILS__
#define __UTILS__

#define INF SHRT_MAX

typedef int32_t** matrix;


matrix allocate_matrix(int32_t size) {
  int32_t i;

  matrix graph_matrix = (int32_t**) malloc (size * sizeof(int32_t*));
  for (i = 0; i < size; i++)
    graph_matrix[i] = (int32_t*) malloc (size * sizeof(int32_t)); 

  return graph_matrix;
}

matrix read_matrix(FILE* graph_file, int32_t size) {
  int32_t i, j;

  matrix graph_matrix = allocate_matrix(size);

  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++) {
    if (!fscanf(graph_file, "%d", &(graph_matrix[i][j]))) {
      printf("Error loading file\n");
      exit(EXIT_FAILURE);
    }
  }
  return graph_matrix;
}

void prepare_matrix(matrix graph_matrix, int32_t size) {
  int i, j;

  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++)
      graph_matrix[i][j] = (i != j && graph_matrix[i][j] == 0) ? INF : graph_matrix[i][j];
}

void print_matrix(matrix distance_matrix, int32_t size) {
  int32_t i, j;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (distance_matrix[i][j] == INF)
        printf("%5s", "INF");
      else
        printf("%5d", distance_matrix[i][j]);
    }
    printf("\n");
  }
}

void free_matrix(matrix matrix, int32_t size) {
  int32_t i;

  for (i = 0; i < size; i++)
    free(matrix[i]);
  free(matrix);
}

#endif
