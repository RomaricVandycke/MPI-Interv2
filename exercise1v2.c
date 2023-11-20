#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Datatype T_triangle;
    int blocklengths[N];
    int displacements[N];

    for (int i = 0; i < N; i++) {
        blocklengths[i] = N - i;
        displacements[i] = (N + 1) * i;
    }

    MPI_Type_indexed(N, blocklengths, displacements, MPI_FLOAT, &T_triangle);
    MPI_Type_commit(&T_triangle);

    if (pid == 0) {
        // Process 0 creates the matrix as a one-dimensional array
        float* A = (float*)malloc(N * N * sizeof(float));
        for (int i = 0; i < N * N; i++) {
            A[i] = (float)(i + 1);
        }

        // Send the lower triangular part with a single call to MPI_Send
        MPI_Send(A, 1, T_triangle, 1, 0, MPI_COMM_WORLD);

        free(A);  // Free allocated memory
    } else if (pid == 1) {
        // Process 1 receives the lower triangular part
        float T[N][N];

        MPI_Recv(T, 1, T_triangle, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Print the received data
        printf("Received lower triangular part:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%f ", T[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&T_triangle);
    MPI_Finalize();
    return 0;
}
