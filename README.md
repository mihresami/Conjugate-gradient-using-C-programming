## General

Sequential and Parallel Conjugate Gradient method using MPI


## Compilation

`make clean` removes the compiled executable `cg`.

`make` compiles the code.

## Running

`mpirun -np <p> ./cg <N> <max_steps>`, where `<p>` is the desired number of processes,
    `<N>` is the problem size - i.e. the N in the NxN matrix generated,
    `<max_steps>` is the max number of steps of conjugate gradient method.
