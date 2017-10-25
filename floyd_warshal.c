#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

#define INF SHRT_MAX


int32_t **read_matrix(FILE* graph_file, int32_t size) {
  int32_t i, j;

  // allocate graph matrix
  int32_t **graph_matrix = (int32_t**) malloc (size * sizeof(int32_t*));
  for (i = 0; i < size; i++)
    graph_matrix[i] = (int32_t*) malloc (size * sizeof(int32_t));

  // load adjacency matrix into the memory
  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++) {
    if (!fscanf(graph_file, "%d", &(graph_matrix[i][j]))) {
      printf("Error loading file\n");
      exit(EXIT_FAILURE);
    }
  }
  return graph_matrix;
}

void print_result(int32_t **distance_matrix, int32_t size) {
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

void prepare_matrix(int32_t **graph_matrix, int32_t size) {
  int i, j;

  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++)
      graph_matrix[i][j] = (i != j && graph_matrix[i][j] == 0) ? INF : graph_matrix[i][j];
}

void floyd_warshall(int32_t **graph_matrix, int32_t size) {
  int32_t i, j, k;

  for (k = 0; k < size; k++)
  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++) {
    if (graph_matrix[i][k] + graph_matrix[k][j] < graph_matrix[i][j])
      graph_matrix[i][j] = graph_matrix[i][k] + graph_matrix[k][j];
  }
}

void free_matrix(int32_t **matrix, int32_t size) {
  int32_t i;

  for (i = 0; i < size; i++)
    free(matrix[i]);
  free(matrix);
}

int main(int argc, char* argv[]) {
  FILE *graph_file;
  int32_t **graph_matrix, size;

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
  prepare_matrix(graph_matrix, size);
  floyd_warshall(graph_matrix, size);

  print_result(graph_matrix, size);

  fclose(graph_file);
  free_matrix(graph_matrix, size);

  return EXIT_SUCCESS;
}
