CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

## Setup Notes:

I modified stream_compaction/CMakeLists.txt as [described here](https://edstem.org/us/courses/81464/discussion/6937028?answer=16157632); adding `find_package(CCCL REQUIRED)` and `target_link_libraries(stream_compaction CCCL::Thrust)` to fix the thrust library not properly being included.


## Questions

# TODO plot, analysis of phenomena
# TODO Nsight analysis

## Program Output:

I didn't add any additional tests to the output here, but I did test compact with arrays not ending in 0, whereas the base test always ended them with 0--this was significant as the use of exclusive scan means the final value in the indices array is equal to the total count of non-zero elements if the final element is 0, but otherwise is 1 less than the total count, so I needed to make sure my implementation handled this case correctly (but since it doesn't particularly affect performance, I did not take separate metrics for it for it).

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
