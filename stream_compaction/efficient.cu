#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


#define DYNAMIC_BLOCK_SIZE 0
// tried out an approach which in addition to reducing the # of blocks when the number of threads needed is small would reduce the size of the one remaining block when the amount of threads is less than that needs, but doesn't seem like it particularly benefits. Leaving in as an option but not using

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernUpSweep(int n, int dExpo, int* data/*, const int* idata*/) {
            // TODO probably worth trying out shared memory way--I guess would move loop into here w/ syncs then change external loop to just when needing to cross boundaries
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return; // sounds like this way is generally better (more explicit that thread can stop)
            }
            int d2 = dExpo << 1;
            int k = (index) * d2;
            data[k + d2 - 1] += data[k + dExpo - 1];

        }
        __global__ void kernDownSweep(int n, int dExpo, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int d2 = dExpo << 1;
            int k = (index)*d2;

            int t = data[k + dExpo - 1];
            data[k + dExpo - 1] = data[k + d2 - 1];
            data[k + d2 - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 32; // Seems to perform best here for 32 or 64 but the change is generally pretty minor
            int dTarget = ilog2ceil(n);
            int pow2Size = 1 << dTarget;
            int nCap = pow2Size >> 1;
            dim3 fullBlocksPerGrid((nCap + blockSize - 1) / blockSize);
            // TODO I'm still not sure if getting fullBlocksPerGrid right 100%, might be overshooting
            int* dev_idata;

            // TODO I think want to rewrite into using shared memory way but that's extra credit so I think don't need to
            //   moving on for now but plan to revisit and do that
            //   also not sure if I've already done what part 5 is referring to? since I don't do modulo and such and decrease # of threads

            cudaMalloc((void**)&dev_idata, pow2Size * sizeof(int));
            
            cudaMemset(dev_idata + n, 0, sizeof(int) * (pow2Size - n));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice); 

#if DYNAMIC_BLOCK_SIZE
            int blockSize2 = blockSize;
#endif

            timer().startGpuTimer();

            // TODO
            
            // up-sweep
            int dExpo = 1; // = 2^(d)
            
            for (int d = 0; d < dTarget; ++d) {
#if DYNAMIC_BLOCK_SIZE
                kernUpSweep<<<fullBlocksPerGrid, blockSize2 >>> (nCap, dExpo, dev_idata);
#else
                kernUpSweep<<<fullBlocksPerGrid, blockSize >>> (nCap, dExpo, dev_idata);
#endif
                if (d < dTarget - 1) {
                    dExpo <<= 1;
                    nCap >>= 1;
                    fullBlocksPerGrid.x = ((nCap + blockSize - 1) / blockSize); // Not sure this is totally the best way to set this but does massively reduce runtime (e.g. ~6ms to ~2ms)

#if DYNAMIC_BLOCK_SIZE
                    // could reduce blockSize when gets really low but not sure worth the effort
                    //  doesn't seem to really matter, so just turning off but making an option anyway
                    if (nCap < blockSize) {
                        blockSize2 = nCap;
                        //blockSize2 = ((nCap - 1) / 32 + 1) * 32;
                    }
                    else {
                        blockSize2 = blockSize;
                    }
#endif
                }
            }
            // set last element to 0
            cudaMemset(dev_idata + (pow2Size - 1), 0, sizeof(int)); 
            // down-sweep
            for (int d = dTarget - 1; d >= 0; --d) {
#if DYNAMIC_BLOCK_SIZE
                if (nCap < blockSize) {
                    // 1 block with block size = # elements used
                    blockSize2 = nCap;
                }
                else {
                    blockSize2 = blockSize;
                }
                kernDownSweep<<<fullBlocksPerGrid, blockSize2>>>(nCap, dExpo, dev_idata);
#else
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(nCap, dExpo, dev_idata);
#endif
                if (d > 0) {
                    dExpo >>= 1;
                    nCap <<= 1;
                    fullBlocksPerGrid.x = ((nCap + blockSize - 1) / blockSize);
                }
            }


            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {


            //int blockSize = testBlockSize; // TODO optimize
            int blockSize = 512; 
            //512 seems to give best performance out of the values tested, but then scan on its own seems best at 32, so using 32 for scan and 512 for rest--seems to give better performance than same size for all the steps
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_idata;
            int* dev_odata;
            int* dev_boolArray;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            int scanBlockSize = 32;
            int dTarget = ilog2ceil(n);
            int pow2Size = 1 << dTarget;
            int nCap = pow2Size >> 1;
            dim3 fullBlocksPerGridScan((nCap + scanBlockSize - 1) / scanBlockSize);

            cudaMalloc((void**)&dev_boolArray, pow2Size * sizeof(int));
            cudaMalloc((void**)&dev_indices, pow2Size * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(dev_odata, 0, sizeof(int) * n);
            cudaMemset(dev_boolArray + n, 0, sizeof(int) * (pow2Size - n));
            




            timer().startGpuTimer();
            // TODO

            // map
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize >>>(n, dev_boolArray, dev_idata);

            // scan
            cudaMemcpy(dev_indices, dev_boolArray, n * sizeof(int), cudaMemcpyDeviceToDevice);
            // up-sweep
            int dExpo = 1; // = 2^(d)

            for (int d = 0; d < dTarget; ++d) {

                kernUpSweep << <fullBlocksPerGridScan, scanBlockSize >> > (nCap, dExpo, dev_indices);

                if (d < dTarget - 1) {
                    dExpo <<= 1;
                    nCap >>= 1;
                    fullBlocksPerGridScan.x = ((nCap + scanBlockSize - 1) / scanBlockSize); // Not sure this is totally the best way to set this but does massively reduce runtime (e.g. ~6ms to ~2ms)
                }

            }
            // set last element to 0
            cudaMemset(dev_indices + (pow2Size - 1), 0, sizeof(int));
            // down-sweep
            for (int d = dTarget - 1; d >= 0; --d) {
                kernDownSweep << <fullBlocksPerGridScan, scanBlockSize >> > (nCap, dExpo, dev_indices);
                if (d > 0) {
                    dExpo >>= 1;
                    nCap <<= 1;
                    fullBlocksPerGridScan.x = ((nCap + scanBlockSize - 1) / scanBlockSize);
                }
            }




            // scatter
            Common::kernScatter<<<fullBlocksPerGrid, blockSize >>>(n, dev_odata, dev_idata, dev_boolArray, dev_indices);

            


            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int count;
            int countStep;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&countStep, dev_boolArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_boolArray);
            cudaFree(dev_indices);

            
            return count + countStep;
        }
    }
}
