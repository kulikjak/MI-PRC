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
            if(!fscanf(graph_file, "%d", &(graph_matrix[i][j]))) {
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

int32_t *dijkstra(int32_t **distance_matrix, int32_t from, int32_t size){

    int32_t *distance = (int32_t*) malloc (size * sizeof(int32_t));

    int pred[size];
    int visited[size],count,mindistance,nextnode;

    //pred[] stores the predecessor of each node
    //count gives the number of nodes seen so far

    //initialize pred[],distance[] and visited[]
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

        //nextnode gives the node at minimum distance
        for(int i=0;i<size;i++)
            if(distance[i]<mindistance&&!visited[i])
            {
                mindistance=distance[i];
                nextnode=i;
            }

        //check if a better path exists through nextnode
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


int32_t **dijkstra_all(int32_t **graph_matrix, int32_t size) {
    int32_t i, j, k;

    // allocate distance matrix
    int32_t **distance_matrix = (int32_t**) malloc (size * sizeof(int32_t*));
    for (i = 0; i < size; i++)
        distance_matrix[i] = (int32_t*) malloc (size * sizeof(int32_t));

    // set default values for distance matrix
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            distance_matrix[i][j] = (i == j) ? 0 : (graph_matrix[i][j]) ? 1 : INF;

    for (int i = 0; i < size; i++)
        distance_matrix[i] = dijkstra(distance_matrix, i, size);

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

    distance_matrix = dijkstra_all(graph_matrix, size);
    print_result(distance_matrix, size);

    free_matrix(distance_matrix, size);
    free_matrix(graph_matrix, size);

    return EXIT_SUCCESS;
}
