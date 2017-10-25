#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

#include "utils.h"


void floyd_warshall(matrix graph_matrix, int32_t size) {
  int32_t i, j, k;

  for (k = 0; k < size; k++)
  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++) {
    if (graph_matrix[i][k] + graph_matrix[k][j] < graph_matrix[i][j])
      graph_matrix[i][j] = graph_matrix[i][k] + graph_matrix[k][j];
  }
}

int main(int argc, char* argv[]) {
  int32_t size;

  FILE *graph_file;
  matrix graph_matrix;

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

  print_matrix(graph_matrix, size);

  fclose(graph_file);
  free_matrix(graph_matrix, size);

  return EXIT_SUCCESS;
}
