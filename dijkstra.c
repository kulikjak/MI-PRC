#include <stdlib.h>
#include <stdio.h>

#include "utils.h"


int32_t *dijkstra(matrix distance_matrix, int32_t from, int32_t size) {

  int32_t *distance = (int32_t*) malloc (size * sizeof(int32_t));

  int pred[size];
  int visited[size],count,mindistance,nextnode;

  for(int i=0;i<size;i++)
  {
    distance[i]=distance_matrix[from][i];
    pred[i]=from;
    visited[i]=0;
  }

  distance[from]=0;
  visited[from]=1;
  count=1;

  while(count<size-1)
  {
    mindistance=INF;

    for(int i=0;i<size;i++)
      if(distance[i]<mindistance&&!visited[i])
      {
        mindistance=distance[i];
        nextnode=i;
      }

    visited[nextnode]=1;
    for(int i=0;i<size;i++)
      if(!visited[i])
        if(mindistance+distance_matrix[nextnode][i]<distance[i])
        {
          distance[i]=mindistance+distance_matrix[nextnode][i];
          pred[i]=nextnode;
        }
    count++;
  }
  return distance;
}

void dijkstra_all(matrix graph_matrix, int32_t size) {
  int32_t i;

  matrix distance_matrix = get_distance_matrix(graph_matrix, size);

  for (i = 0; i < size; i++)
    graph_matrix[i] = dijkstra(distance_matrix, i, size);
}

int main(int argc, char* argv[]) {
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
  fclose(graph_file);
  distance_matrix dijkstra_all(graph_matrix, size);

  print_matrix(distance_matrix, size);

  free_matrix(distance_matrix, size);
  free_matrix(graph_matrix, size);

  return EXIT_SUCCESS;
}
