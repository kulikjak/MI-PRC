CXX = gcc -g -O2

all: floyd_warshal dijkstra

floyd_warshal: floyd_warshal.c utils.h
	$(CXX) $^ -o $@

dijkstra: dijkstra.c utils.h
	$(CXX) $^ -o $@

randomizer: randomizer.c utils.h
	$(CXX) $^ -o $@

clean:
	rm -f floyd_warshal dijkstra randomizer

.PHONY: all floyd_warshal dijkstra randomizer clean
