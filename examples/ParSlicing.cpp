#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);

  const int mBegin = argc > 1 ? atol(argv[1]) : 1;
  const int mEnd = argc > 2 ? atol(argv[2]) : 10;
  const int maxComputeCount = argc > 3 ? atol(argv[3]) : 1;

  int rank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (nprocs < 2) {
    printf("Error: number of processes has to be larger than 1\n");
    exit(EXIT_FAILURE);
  }

  struct Task {
    int size;
    int begin;
    int end;
  };
  const int TAG_TASK = 0;
  const int TAG_SPLIT_TASK = 1;
  const int TAG_RESULT = 2;
  const int TAG_FINISH = 3;
  MPI_Status status;
  bool isFinished;
  int resultCount;
  double elapsedTime = 0;
  std::vector<int> incomingResult;
  std::vector<int> taskBuffer(6);
  std::vector<int> resultBuffer(2 * maxComputeCount);

  MPI_Barrier(MPI_COMM_WORLD);
  elapsedTime -= MPI_Wtime();

  if (rank == 0) {  // Master
    std::queue<Task> pool;
    std::vector<bool> isIdle(nprocs, true);
    isIdle[rank] = false;  // Set master node as working

    const int targetCount = mEnd - mBegin + 1;
    pool.push(Task{targetCount, mBegin, mEnd});
    int finishedCount = 0;
    // printf("Master: Insert task (%d, %d, %d) into pool\n",
    //        targetCount, mBegin, mEnd);
    while (finishedCount < targetCount) {
      // TODO Optimize using bitset
      for (int i = 1; (i < nprocs) && (!pool.empty()); i++) {
        if (isIdle[i]) {
          // Send task to process number i
          const auto task = pool.front();
          taskBuffer[0] = task.size;
          taskBuffer[1] = task.begin;
          taskBuffer[2] = task.end;

          // printf("Master: Send task (%d, %d, %d) to Slave-%d\n",
          //        task.size, task.begin, task.end, i);
          MPI_Send(taskBuffer.data(), 3, MPI_INT, i, TAG_TASK, MPI_COMM_WORLD);
          pool.pop();
          isIdle[i] = false;
        }
      }
      // Receive message from slave
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case TAG_SPLIT_TASK: {
          // Receive 6 numbers denoting two new tasks
          // printf("Master: Receive split task from Slave-%d\n", status.MPI_SOURCE);
          MPI_Recv(taskBuffer.data(), 6, MPI_INT,
                   status.MPI_SOURCE, TAG_SPLIT_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // Insert two new tasks into pool
          Task taskLeft  { taskBuffer[0], taskBuffer[1], taskBuffer[2] };
          Task taskRight { taskBuffer[3], taskBuffer[4], taskBuffer[5] };
          if (taskLeft.size > 0) {
            // printf("Master: Insert task (%d, %d, %d) into pool\n",
            //        taskLeft.size, taskLeft.begin, taskLeft.end);
            pool.push(taskLeft);
          }
          if (taskRight.size > 0) {
            // printf("Master: Insert task (%d, %d, %d) into pool\n",
            //        taskRight.size, taskRight.begin, taskRight.end);
            pool.push(taskRight);
          }
          break;
        }
        case TAG_RESULT: {
          // Receive a number of results from slave
          MPI_Get_count(&status, MPI_INT, &resultCount);
          MPI_Recv(resultBuffer.data(), resultCount, MPI_INT,
                   status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // char outBuffer[1024];
          // int offset = 0;
          // int charsWritten;
          // for (int j = 0; j < resultCount; j++) {
          //   charsWritten = snprintf(outBuffer+offset, 1024-offset, "%d ", resultBuffer[j]);
          //   offset += charsWritten;
          //   incomingResult.push_back(resultBuffer[j]);
          // }
          // printf("Master: Receive %d results=[ %s] from Slave-%d\n",
          //        resultCount, outBuffer, status.MPI_SOURCE);

          finishedCount += resultCount;
          isIdle[status.MPI_SOURCE] = true;
          break;
        }
      }
    }
    // printf("Master: All %d results have been received\n", finishedCount);
    // Send finish signal to all process
    isFinished = true;
    for (int i = 1; i < nprocs; i++) {
      // printf("Master: Send finish signal to Slave-%d\n", i);
      MPI_Send(&isFinished, 1, MPI_C_BOOL, i, TAG_FINISH, MPI_COMM_WORLD);
    }
    // printf("Master: Finished\n");
  }
  else {  // Slave
    isFinished = false;
    while(!isFinished) {
      // Wait for message from master
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case TAG_TASK: {
          // Receive task from master
          MPI_Recv(taskBuffer.data(), 3, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // printf("Slave-%d: Receive task (%d, %d, %d) from Master\n",
          //        rank, taskBuffer[0], taskBuffer[1], taskBuffer[2]);
          Task task {taskBuffer[0], taskBuffer[1], taskBuffer[2]};
          if (task.size > 2 * maxComputeCount) {
            // Compute only maxComputeCount
            const auto computeBegin = task.begin + (task.size / 2) - (maxComputeCount / 2);
            const auto computeEnd = computeBegin + maxComputeCount - 1;
            // Split into two tasks
            Task taskLeft  { (computeBegin - 1) - task.begin + 1, task.begin, computeBegin - 1 };
            Task taskRight { task.end - (computeEnd + 1) + 1, computeEnd + 1, task.end };
            taskBuffer[0] = taskLeft.size;
            taskBuffer[1] = taskLeft.begin;
            taskBuffer[2] = taskLeft.end;
            taskBuffer[3] = taskRight.size;
            taskBuffer[4] = taskRight.begin;
            taskBuffer[5] = taskRight.end;
            // Update current task
            task.begin = computeBegin;
            task.end = computeEnd;
            task.size = task.end - task.begin + 1;
            // Send split task to master
            // printf("Slave-%d: Send split task (%d, %d, %d) and (%d, %d, %d) to Master\n",
            //        rank, taskBuffer[0], taskBuffer[1], taskBuffer[2],
            //        taskBuffer[3], taskBuffer[4], taskBuffer[5]);
            MPI_Send(taskBuffer.data(), 6, MPI_INT, 0, TAG_SPLIT_TASK, MPI_COMM_WORLD);
          }
          // Perform current task
          resultCount = task.size;
          for (int i = task.begin; i <= task.end; i++) {
            resultBuffer[i - task.begin] = i;
          }

          // char outBuffer[1024];
          // int offset = 0;
          // int charsWritten;
          // for (int i = 0; i < task.size; i++) {
          //   charsWritten = snprintf(outBuffer+offset, 1024-offset, "%d ", resultBuffer[i]);
          //   offset += charsWritten;
          // }
          // printf("Slave-%d: Send %d results=[ %s] to Master\n", rank, resultCount, outBuffer);
          MPI_Send(resultBuffer.data(), resultCount, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);

          break;
        }
        case TAG_FINISH: {
          // Receive finish signal
          // printf("Slave-%d: Receive finish signal from Master\n", rank);
          MPI_Recv(&isFinished, 1, MPI_C_BOOL, 0, TAG_FINISH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          break;
        }
      }
    }
    // printf("Slave-%d: Finished\n", rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  elapsedTime += MPI_Wtime();

  if (rank == 0) {
    printf("Time: %.2lf secs\n", elapsedTime);
  }

  MPI_Finalize();
  return 0;
}
