#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

double _start_in, _start_out;
double _end_in, _end_out;

matrix dijkstra_all(matrix distance_matrix, int32_t size) {
  int32_t from;

  #pragma acc data copy(distance_matrix[0 : size][0 : size])
  {
    _start_in = omp_get_wtime();
    #pragma acc parallel
    {
      #pragma acc loop independent
      for (from = 0; from < size; from++) {
        int32_t *visited = (int32_t *)malloc(size * sizeof(int32_t));
        int32_t i, count, mindistance, nextnode;

        #pragma acc loop
        for (i = 0; i < size; visited[i++] = 0) {}

        visited[from] = 1;

        #pragma acc loop
        for (count = 1; count < size - 1; count++) {
          mindistance = INF;

          #pragma acc loop
          for (i = 0; i < size; i++) {
            if (distance_matrix[from][i] < mindistance && !visited[i]) {
              mindistance = distance_matrix[from][i];
              nextnode = i;
            }
          }

          visited[nextnode] = 1;

          #pragma acc loop
          for (i = 0; i < size; i++) {
            if (!visited[i]) {
              if (mindistance + distance_matrix[nextnode][i] < distance_matrix[from][i]) {
                #pragma acc atomic write
                distance_matrix[from][i] = mindistance + distance_matrix[nextnode][i];
              }
            }
          }
        }
        free(visited);
      }
    }
    _end_in = omp_get_wtime();
  }
  return distance_matrix;
}

int main(int argc, char *argv[]) {
  FILE *graph_file;

  int32_t size;
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
  distance_matrix = get_distance_matrix(graph_matrix, size);

  _start_out = omp_get_wtime();  // clock();
  distance_matrix = dijkstra_all(distance_matrix, size);
  _end_out = omp_get_wtime();  // clock();

  // print_matrix(distance_matrix, size);

  free_matrix(distance_matrix, size);
  free_matrix(graph_matrix, size);
  fclose(graph_file);

  printf("== Time: %lf (without data copy)\n", _end_in - _start_in);
  printf("== Time: %lf (with data copy)\n", _end_out - _start_out);

  return EXIT_SUCCESS;
}
