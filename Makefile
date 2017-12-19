CXX = gcc -fopenmp -O2
PGXX = pgcc -acc -ta=nvidia -Minfo=acc -O0
NVXX = nvcc -gencode=arch=compute_30,code=compute_30 -Xptxas -v

all: floyd_warshall dijkstra

acc: acc_floyd_warshall acc_dijkstra

cuda: cuda_floyd_warshall cuda_dijkstra



fw: floyd_warshall

acc_fw: acc_floyd_warshall

cuda_fw: cuda_floyd_warshall


fw_kernels: cuda_fw_0 cuda_fw_1 cuda_fw_2 cuda_fw_3

cuda_fw_0: floyd_warshall.cu cuda_utils.h
	$(NVXX) floyd_warshall.cu -D _KERNEL=0 -o $@

cuda_fw_1: floyd_warshall.cu cuda_utils.h
	$(NVXX) floyd_warshall.cu -D _KERNEL=1 -o $@

cuda_fw_2: floyd_warshall.cu cuda_utils.h
	$(NVXX) floyd_warshall.cu -D _KERNEL=2 -o $@

cuda_fw_3: floyd_warshall.cu cuda_utils.h
	$(NVXX) floyd_warshall.cu -D _KERNEL=3 -o $@


floyd_warshall: floyd_warshall.c utils.h
	$(CXX) floyd_warshall.c -o $@

acc_floyd_warshall: floyd_warshall.c utils.h
	$(PGXX) floyd_warshall.c -o $@

cuda_floyd_warshall: floyd_warshall.cu cuda_utils.h
	$(NVXX) floyd_warshall.cu -o $@

dijkstra: dijkstra.c utils.h
	$(CXX) dijkstra.c -o $@

acc_dijkstra: dijkstra.c utils.h
	$(PGXX) dijkstra.c -o $@

cuda_dijkstra: dijkstra.cu cuda_utils.h
	$(NVXX) dijkstra.cu -o $@

clean:
	rm -f floyd_warshall dijkstra acc_floyd_warshall acc_dijkstra cuda_floyd_warshall cuda_dijkstra
	rm -f *.dwf *.pdb

.PHONY: all clean
