CXX = gcc -O2

all: floyd_warshal dijkstra

floyd_warshal: floyd_warshal.c
	$(CXX) $^ -o $@

dijkstra: dijkstra.c
	$(CXX) $^ -o $@

clean:
	rm -f floyd_warshal
	rm -f dijkstra

.PHONY: all floyd_warshal dijkstra clean
