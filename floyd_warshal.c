#include <stdlib.h>
#include <stdio.h>

#define INF 65536


int32_t **read_matrix(FILE* graph_file, int32_t size) {
  int32_t i, j;

  // allocate graph matrix
  int32_t **graph_matrix = (int32_t**) malloc (size * sizeof(int32_t*));
  for (i = 0; i < size; i++)
    graph_matrix[i] = (int32_t*) malloc (size * sizeof(int32_t));

  // load adjacency matrix into the memory
  for (i = 0; i < size; i++)
  for (j = 0; j < size; j++)
    fscanf(graph_file, "%d", &(graph_matrix[i][j]));

  return graph_matrix;
}

void print_result(int32_t **distance_matrix, int32_t size) {
  int32_t i, j;

  printf("Shortest distances between every pair of vertices: \n");

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

int32_t **floyd_warshall(int32_t **graph_matrix, int32_t size) {
  int32_t i, j, k;

  // allocate distance matrix
  int32_t **distance_matrix = (int32_t**) malloc (size * sizeof(int32_t*));
  for (i = 0; i < size; i++)
    distance_matrix[i] = (int32_t*) malloc (size * sizeof(int32_t));

  // set default values for distance matrix
  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++)
      distance_matrix[i][j] = graph_matrix[i][j];

  // run floyd warshal itself
  for (int k = 0; k < size; k++)
  for (int i = 0; i < size; i++)
  for (int j = 0; j < size; j++) {
    if (distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j])
      distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j];
  }

  return distance_matrix;
}

void free_matrix(int32_t **matrix, int32_t size) {
  int32_t i;

  for (i = 0; i < size; i++)
    free(matrix[i]);
  free(matrix);
}

int main(int argc, char* argv[]) {
  FILE *graph_file;
  int32_t size;
  int32_t **distance_matrix, **graph_matrix;

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
  fclose(graph_file);

  distance_matrix = floyd_warshall(graph_matrix, size);
  print_result(distance_matrix, size);

  free_matrix(distance_matrix, size);
  free_matrix(graph_matrix, size);

  return EXIT_SUCCESS;
}
