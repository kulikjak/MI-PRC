#include <stdlib.h>
#include <stdio.h>

#include "utils.h"


matrix floyd_warshall(const matrix graph_matrix, int32_t size) {
  int32_t i, j, k;

  matrix distance_matrix = get_distance_matrix(graph_matrix, size);

  for (k = 0; k < size; k++)
  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++) {
    if (distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j])
      distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j];
  }

  return distance_matrix;
}

int main(int argc, char* argv[]) {
  int32_t size;

  FILE *graph_file;
  matrix graph_matrix, distance_matrix;

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
  distance_matrix = floyd_warshall(graph_matrix, size);

  print_matrix(distance_matrix, size);

  free_matrix(graph_matrix, size);
  free_matrix(distance_matrix, size);
  fclose(graph_file);

  return EXIT_SUCCESS;
}
