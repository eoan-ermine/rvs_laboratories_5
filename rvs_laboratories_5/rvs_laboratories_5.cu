#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Вычисление C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float result = 0;

  if (row < numARows && col < numBColumns)
  {
    for (int i = 0; i < numAColumns; ++i)
    {
      result += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    C[row * numCColumns + col] = result;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // Матрица A
  float *hostB; // Матрица B
  float *hostC; // Выходная матрица C
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // количество строк матрицы A
  int numAColumns; // количество столбцов матрицы A
  int numBRows;    // количество строк матрицы B
  int numBColumns; // количество столбцов матрицы B
  int numCRows;    // количество строк матрицы  C (установите
                              // это значение сами)
  int numCColumns; // количество столбцов матрицы C (установите
                   //это значение сами)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Установите numCRows и numCColumns
  numCRows    = numARows;
  numCColumns = numAColumns;
  //@@ Выделение памяти под матрицу hostC
  hostC = static_cast<float*>(malloc(numCRows * numCColumns * sizeof(float)));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Выделите память GPU
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Скопируйте память с хоста на GPU
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Инициализируйте размерности блоков и сетки

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ запустите ядро GPU


  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Скопируйте память обратно с GPU на хост
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Освободите память GPU
  cudaFree(deviceA);
  cudaFree(deviceB);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
