#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

#include "utils.h"


#define _RAND_MIN 1
#define _RAND_MAX 10


void randomize_matrix(matrix graph_matrix, int32_t size) {
  int32_t i, j, r;

  for (i = 0; i < size; i++)
  for (j = i+1; j < size; j++) {
    if (graph_matrix[i][j]) {
      r = (rand() % (_RAND_MAX + 1 - _RAND_MIN)) + _RAND_MIN;
      graph_matrix[i][j] = r;
      graph_matrix[j][i] = r;
    }
  }
}

void save_matrix(FILE* output_file, matrix matrix, int32_t size) {
  int32_t i, j;

  fprintf(output_file, "%d\n", size);
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++)
      fprintf(output_file, "%d ", matrix[i][j]);
    fprintf(output_file, "\n");
  }
}


int main(int argc, char* argv[]) {
  FILE *graph_file, *output_file;
  matrix graph_matrix;
  int32_t size;

  if (argc != 3) {
    printf("Wrong input\n");
    printf("Usage: %s GRAPH_FILE OUTPUT_FILE\n", argv[0]);
    return EXIT_FAILURE;
  }

  graph_file = fopen(argv[1], "r");
  if (graph_file == NULL) {
    printf("Cannot open input file.\n");
    return EXIT_FAILURE;
  }

  output_file = fopen(argv[2], "w");
  if (graph_file == NULL) {
    printf("Cannot open output file.\n");
    return EXIT_FAILURE;
  }

  if (fscanf(graph_file, "%d", &size) == EOF) {
    printf("Input file is empty.\n");
    return EXIT_FAILURE;
  }

  graph_matrix = read_matrix(graph_file, size);

  randomize_matrix(graph_matrix, size);
  save_matrix(output_file, graph_matrix, size);

  free_matrix(graph_matrix, size);

  fclose(graph_file);
  fclose(output_file);

  return EXIT_SUCCESS;
}
