CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Griffin Evans
  * gpevans@seas.upenn.edu, [personal website](evanses.com/griffin)
* Tested on lab computer: Windows 11 Education, i9-12900F @ 2.40GHz 64.0GB, NVIDIA GeForce RTX 3090 (Levine 057 #1)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

## Setup Notes:

I modified stream_compaction/CMakeLists.txt as [described here](https://edstem.org/us/courses/81464/discussion/6937028?answer=16157632); adding `find_package(CCCL REQUIRED)` and `target_link_libraries(stream_compaction CCCL::Thrust)` to fix the thrust library not properly being included.

TODO not sure order to put this in:
## Work-efficient Parallel Scan Implementation (Part 5)

In order to ensure the work-efficient implementation of scan was performant, a few considerations were made. Firstly note that in both the up-sweep and down-sweep kernels, in order to set the $k$ value (used in the indices of elements to be summed) which ranges from $0$ to $n-1$ (where $n$ is the number of elements in the padded array) in steps of $2^{d+1}$, we take the index of the thread (calculated from `threadIdx`, `blockIdx`, and `blockDim`), compare it to a maximum value and if it is greater than or equal to that value then we return, and if it isn't then we scale that index by multiplying it by $2^{d+1}$ and use that as our $k$.

For the up-sweep, since the step size $2^{d+1}$ doubles in each iteration, we halve this maximum value each time, starting at $n/2$ as the first step size is $2^{0+1} = 2$. This means in each iteration we have half the number of threads which need to perform work, and the rest of the threads can terminate early upon reaching that maximum index check. Because all of the threads that have work required are sequential in index (their indices starting at 0 and going up by 1 to our maximum index at that step) we should have at most one warp which will have some threads working while others try to terminate early, while all other warps either have all threads doing work or all threads terminating early. In fact since we know the higher-index threads will not have to do any work, we can reduce the number of blocks launched as we reduce this maximum index, such that we completely skip any blocks that would have no threads which actually perform work.

For the down-sweep, as $d$ is now descending and $2^{d+1}$ halves each time, we instead double the maximum value each time, such that the number of threads needing to do work doubles each time, and we similarly increase the number of blocks launched as the number of threads needed increases.

An additional detail which was tested but ultimately is unused in this implementation was the reduction of size of the block once the number of threads needed to launch was low enough that we only had a single block. That is, when the number of threads that will do work is less than the default block size, the kernel would launch with a single block with a number of threads equal to the number of summations that will be done (also tested with that value rounded up to the next multiple of 32, to correspond with the size of warps). This however did not appear to cause any noticeable change in performance, and in testing various default block sizes it was found that starting all of the blocks with 32 threads (the size of a single warp) was most performant for running this implementation of scan on the hardware used&mdash;so since the blocks were single-warp-sized anyway (meaning there'd be no particular benefit to shrinking a block further, since 32 threads would make up the warp anyway), it was not necessary to use this size reduction behavior.

## Questions

# TODO plot, analysis of phenomena

For smaller array sizes, the CPU implementation is plenty efficient and outperforms both the Naive and Work-efficient implementations

# TODO Nsight analysis

![](img/Overview1.png)
![](img/Overview2.png)

Looking at the timeline in NSight Systems, we see a few main periods of activity that seem to be associated with the calls to the Thrust library&mdash;firstly corresponding to the construction of the device vectors, then to the scan function itself, then to the copy operation sending the resulting data to the CPU side.

![](img/Zoom1.png)

TODO analysis TODO not actually sure if this or the earlier part is the main setup of vectors TODO I think this is the synching at end of vector setup, need to retake zoomed out more

![](img/Zoom2.png)

This span corresponds to the time measured with `timer().startGpuTimer()`&mdash;that is, the execution of `thrust::exclusive_scan` itself. 

![](img/Zoom3.png)

TODO

For all of the implementations, initial and final memory operations (primarily calls to cudaMemcpy) take take longer than the execution of the algorithm itself, hence being left out of our timing comparisons.

As the naive implementation here does not vary in number of threads between steps (with threads that aren't summing just copying values from idata to odata, rather than terminating early or not starting at all; the only threads that terminate early are those which have index greater than the total count of elements, i.e. only a few threads in one block when the array size is non-power-of-two), the kernel used by it has a fairly consistent length of execution. For the work-efficient implementation, the kernels vary in length of execution significantly likely because we both have more threads terminate early and we only launch blocks of threads up to the amount required for the work to actually be done in that step. The first calling of the up-sweep kernel takes the longest to complete, then each subsequent step becomes shorter as we reduce the number of threads needed, and inversely the down-sweep kernel starts the shortest then gets longer as we increase the number of threads used.

## Program Output:

I didn't add any additional tests to the output here, but I did test compact with arrays not ending in 0, whereas the base test always ended them with 0&mdash;this was significant as the use of exclusive scan means the final value in the indices array is equal to the total count of non-zero elements if the final element is 0, but otherwise is 1 less than the total count, so I needed to make sure my implementation handled this case correctly (but since it doesn't particularly affect performance, I did not take separate metrics for it for it).

With array size 2^24:
```
Array Size 2^24

****************
** SCAN TESTS **
****************
    [  33  14  48   6  37  31  38  47  49  45   1   8  38 ...  36   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 26.9298ms    (std::chrono Measured)
    [   0  33  47  95 101 138 169 207 254 303 348 349 357 ... 410991594 410991630 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 26.3172ms    (std::chrono Measured)
    [   0  33  47  95 101 138 169 207 254 303 348 349 357 ... 410991573 410991580 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 4.8169ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 4.69709ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 2.28694ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 2.09229ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.608256ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.741376ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   0   2   3   0   2   2   2   1   1   2   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 31.3233ms    (std::chrono Measured)
    [   1   2   2   3   2   2   2   1   1   2   3   1   1 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 32.9821ms    (std::chrono Measured)
    [   1   2   2   3   2   2   2   1   1   2   3   1   1 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 65.5021ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2.65523ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2.58253ms    (CUDA Measured)
    passed
Press any key to continue . . .
```
