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

#include <unistd.h>
#include <mpi.h>

// #define DEBUG_OUTPUT

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);

  const int mBegin = argc > 1 ? atol(argv[1]) : 1;
  const int mEnd = argc > 2 ? atol(argv[2]) : 10;
  const int maxComputeCount = argc > 3 ? atol(argv[3]) : 1;

  int rank, nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (nProcs < 2) {
    printf("Error: number of processes has to be larger than 1\n");
    exit(EXIT_FAILURE);
  }

  struct Task {
    int size;
    int begin;
    int end;
  };
  const int TIME_PER_EV = 100 * 1000;  // microseconds
  const int TAG_TASK = 0;
  const int TAG_SPLIT_TASK = 1;
  const int TAG_RESULT = 2;
  const int TAG_FINISH = 3;
  MPI_Status status;
  bool isFinished;
  int resultCount;
  double elapsedTime = 0;
  std::vector<int> incomingResult;
  std::vector<int> resultBuffer(2 * maxComputeCount);

  MPI_Barrier(MPI_COMM_WORLD);
  elapsedTime -= MPI_Wtime();

  if (rank == 0) {  // Master
    std::queue<Task> taskPool;
    std::vector<bool> isIdle(nProcs, true);
    std::vector<int> inTaskBuffer(6);
    std::vector<int> outTaskBuffer(3 * nProcs);
    std::vector<MPI_Request> outTaskRequests(nProcs-1), finishRequests(nProcs-1);
    isIdle[rank] = false;  // Set master node as working

    const int targetCount = mEnd - mBegin + 1;
    taskPool.push(Task{targetCount, mBegin, mEnd});  // Push initial task
#ifdef DEBUG_OUTPUT
    printf("Master: Inserted task (%d, %d, %d) into taskPool\n",
           targetCount, mBegin, mEnd);
#endif
    int finishedCount = 0;
    while (finishedCount < targetCount) {
      for (int i = 1; (i < nProcs) && (!taskPool.empty()); i++) {
        if (isIdle[i]) {
          // Send task to process number i
          const auto task = taskPool.front();
          outTaskBuffer[i * 3 + 0] = task.size;
          outTaskBuffer[i * 3 + 1] = task.begin;
          outTaskBuffer[i * 3 + 2] = task.end;

          MPI_Isend(outTaskBuffer.data() + i * 3, 3, MPI_INT, i, TAG_TASK,
                    MPI_COMM_WORLD, &outTaskRequests[i-1]);
          MPI_Request_free(&outTaskRequests[i-1]);
          taskPool.pop();
          isIdle[i] = false;
#ifdef DEBUG_OUTPUT
          printf("Master: Sent task (%d, %d, %d) to Slave-%d\n",
                 task.size, task.begin, task.end, i);
#endif
        }
      }
      // Receive message from slave
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case TAG_SPLIT_TASK: {
          // Receive 6 numbers denoting two new tasks
          MPI_Recv(inTaskBuffer.data(), 6, MPI_INT,
                   status.MPI_SOURCE, TAG_SPLIT_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Master: Received split task from Slave-%d\n", status.MPI_SOURCE);
#endif
          // Insert two new tasks into taskPool
          Task taskLeft  { inTaskBuffer[0], inTaskBuffer[1], inTaskBuffer[2] };
          Task taskRight { inTaskBuffer[3], inTaskBuffer[4], inTaskBuffer[5] };
          if (taskLeft.size > 0) {
            taskPool.push(taskLeft);
#ifdef DEBUG_OUTPUT
            printf("Master: Inserted task (%d, %d, %d) into taskPool\n",
                   taskLeft.size, taskLeft.begin, taskLeft.end);
#endif
          }
          if (taskRight.size > 0) {
            taskPool.push(taskRight);
#ifdef DEBUG_OUTPUT
            printf("Master: Inserted task (%d, %d, %d) into taskPool\n",
                   taskRight.size, taskRight.begin, taskRight.end);
#endif
          }
          break;
        }
        case TAG_RESULT: {
          // Receive a number of results from slave
          MPI_Get_count(&status, MPI_INT, &resultCount);
          MPI_Recv(resultBuffer.data(), resultCount, MPI_INT,
                   status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          char outBuffer[1024];
          int offset = 0;
          int charsWritten;
          for (int j = 0; j < resultCount; j++) {
            charsWritten = snprintf(outBuffer+offset, 1024-offset, "%d ", resultBuffer[j]);
            offset += charsWritten;
            incomingResult.push_back(resultBuffer[j]);
          }
          printf("Master: Received %d results=[ %s] from Slave-%d\n",
                 resultCount, outBuffer, status.MPI_SOURCE);
#endif
          finishedCount += resultCount;
          isIdle[status.MPI_SOURCE] = true;
          break;
        }
      }
    }
#ifdef DEBUG_OUTPUT
    printf("Master: All %d results have been received\n", finishedCount);
#endif
    // Send finish signal to all process
    isFinished = true;
    for (int i = 1; i < nProcs; i++) {
      MPI_Isend(&isFinished, 1, MPI_C_BOOL, i, TAG_FINISH, MPI_COMM_WORLD, &finishRequests[i-1]);
      MPI_Request_free(&finishRequests[i-1]);
#ifdef DEBUG_OUTPUT
      printf("Master: Sent finish signal to Slave-%d\n", i);
#endif
    }
#ifdef DEBUG_OUTPUT
    printf("Master: Finished\n");
#endif
  }
  else {  // Slave
    std::vector<MPI_Request> sendRequests(2);
    std::vector<int> inTaskBuffer(3);
    std::vector<int> outTaskBuffer(6);
    bool splitTask;
    isFinished = false;
    while(!isFinished) {
      // Wait for message from master
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case TAG_TASK: {
          // Receive task from master
          MPI_Recv(inTaskBuffer.data(), 3, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Slave-%d: Received task (%d, %d, %d) from Master\n",
                 rank, inTaskBuffer[0], inTaskBuffer[1], inTaskBuffer[2]);
#endif
          Task task {inTaskBuffer[0], inTaskBuffer[1], inTaskBuffer[2]};
          splitTask = task.size > 2 * maxComputeCount;
          if (splitTask) {
            // Compute only maxComputeCount
            const auto computeBegin = task.begin + (task.size / 2) - (maxComputeCount / 2);
            const auto computeEnd = computeBegin + maxComputeCount - 1;
            usleep(2 * TIME_PER_EV);
            // Split into two tasks
            Task taskLeft  { (computeBegin - 1) - task.begin + 1, task.begin, computeBegin - 1 };
            Task taskRight { task.end - (computeEnd + 1) + 1, computeEnd + 1, task.end };
            outTaskBuffer[0] = taskLeft.size;
            outTaskBuffer[1] = taskLeft.begin;
            outTaskBuffer[2] = taskLeft.end;
            outTaskBuffer[3] = taskRight.size;
            outTaskBuffer[4] = taskRight.begin;
            outTaskBuffer[5] = taskRight.end;
            // Update current task
            task.begin = computeBegin;
            task.end = computeEnd;
            task.size = task.end - task.begin + 1;
            // Send split task to master
            MPI_Isend(outTaskBuffer.data(), 6, MPI_INT, 0, TAG_SPLIT_TASK, MPI_COMM_WORLD, &sendRequests[1]);
#ifdef DEBUG_OUTPUT
            printf("Slave-%d: Return split task (%d, %d, %d) and (%d, %d, %d) to Master\n", rank,
                   outTaskBuffer[0], outTaskBuffer[1], outTaskBuffer[2],
                   outTaskBuffer[3], outTaskBuffer[4], outTaskBuffer[5]);
#endif
          }
          // Perform current task
          resultCount = task.size;
          for (int i = task.begin; i <= task.end; i++) {
            resultBuffer[i - task.begin] = i;
            usleep(TIME_PER_EV);
          }
          // Send result of current task
          MPI_Isend(resultBuffer.data(), resultCount, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD, &sendRequests[0]);
#ifdef DEBUG_OUTPUT
          char outBuffer[1024];
          int offset = 0;
          int charsWritten;
          for (int i = 0; i < task.size; i++) {
            charsWritten = snprintf(outBuffer+offset, 1024-offset, "%d ", resultBuffer[i]);
            offset += charsWritten;
          }
          printf("Slave-%d: Sent %d results=[ %s] to Master\n", rank, resultCount, outBuffer);
#endif
          // Wait until all sends are done
          int readyForNextTask = 0;
          while (!readyForNextTask) {
            MPI_Testall(splitTask ? 2 : 1, sendRequests.data(), &readyForNextTask, MPI_STATUS_IGNORE);
          }
          break;
        }
        case TAG_FINISH: {
          // Receive finish signal
          MPI_Recv(&isFinished, 1, MPI_C_BOOL, 0, TAG_FINISH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Slave-%d: Received finish signal from Master\n", rank);
#endif
          break;
        }
      }
    }
#ifdef DEBUG_OUTPUT
    printf("Slave-%d: Finished\n", rank);
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  elapsedTime += MPI_Wtime();

  if (rank == 0) {
#ifdef DEBUG_OUTPUT
    // Print order of results
    char outBuffer[1024];
    int offset = 0;
    int charsWritten;
    for (int i = 0; i < incomingResult.size(); i++) {
      charsWritten = snprintf(outBuffer+offset, 1024-offset, "%d ", incomingResult[i]);
      offset += charsWritten;
    }
    printf("Master: Received %ld results =[ %s]\n", incomingResult.size(), outBuffer);
#endif
    printf("Time: %.2lf secs\n", elapsedTime);
  }

  MPI_Finalize();
  return 0;
}
