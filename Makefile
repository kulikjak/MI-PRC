CXX = gcc -fopenmp -O2
PGXX = pgcc -acc -Minfo=acc -O2

all: floyd_warshal dijkstra

acc: acc_floyd_warshal

floyd_warshal: floyd_warshal.c utils.h
	$(CXX) floyd_warshal.c -o $@

acc_floyd_warshal: floyd_warshal.c utils.h
	$(PGXX) floyd_warshal.c -o $@

dijkstra: dijkstra.c utils.h
	$(CXX) dijkstra.c -o $@

randomizer: randomizer.c utils.h
	$(CXX) $^ -o $@

clean:
	rm -f floyd_warshal dijkstra randomizer acc_floyd_warshal
	rm -f *.dwf *.pdb

.PHONY: all floyd_warshal dijkstra randomizer acc_floyd_warshal clean
