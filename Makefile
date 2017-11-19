CXX = gcc -fopenmp -O2
PGXX = pgcc -acc -Minfo=acc -O2

all: floyd_warshall dijkstra

acc: acc_floyd_warshal

floyd_warshall: floyd_warshall.c utils.h
	$(CXX) floyd_warshall.c -o $@

acc_floyd_warshall: floyd_warshall.c utils.h
	$(PGXX) floyd_warshall.c -o $@

dijkstra: dijkstra.c utils.h
	$(CXX) dijkstra.c -o $@

randomizer: randomizer.c utils.h
	$(CXX) $^ -o $@

clean:
	rm -f floyd_warshall dijkstra randomizer acc_floyd_warshall
	rm -f *.dwf *.pdb

.PHONY: all floyd_warshall dijkstra randomizer acc_floyd_warshall clean
