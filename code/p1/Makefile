# File       : Makefile
# Description: Makefile for peak performance code
# Copyright 2024 Harvard University. All Rights Reserved.
CXX = g++
CPPFLAGS = -g -Wall -Wextra -fopenmp
# TODO: add compiler flags for your kernel here
PERFLAGS = -O3 -ffast-math -mavx2 -fopenmp  # Optimized performance flags
.PHONY: clean

main: main.cpp kernel.o
	$(CXX) $(CPPFLAGS) $(PERFLAGS) -o $@ $^

kernel.o: kernel.cpp
	$(CXX) $(CPPFLAGS) $(PERFLAGS) -c -o $@ $<

clean:
	rm -f main *.o
