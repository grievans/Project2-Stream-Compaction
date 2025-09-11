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

