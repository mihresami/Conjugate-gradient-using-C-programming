CC=mpicc
CFLAGS=-w -O3 -std=c99
LIBS=-lmpi -lm

cg: main.c cg.c
	$(CC) $(CFLAGS) cg.c main.c -o cg $(LIBS)

clean: 
	rm -f cg
