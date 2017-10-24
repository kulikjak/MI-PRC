CXX = gcc -O2

all: floyd_warshal

floyd_warshal: floyd_warshal.c
	$(CXX) $^ -o $@

clean:
	rm -f floyd_warshal

.PHONY: all floyd_warshal clean
