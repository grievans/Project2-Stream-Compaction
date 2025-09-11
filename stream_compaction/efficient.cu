#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernUpSweep(int n, int dExpo, int* data/*, const int* idata*/) {
            // TODO I assume should invoke these with only the number of blocks actually used rather than with all the blocks when most don't do any, but will do following structure more literally first?
            // TODO probably worth trying out shared memory way--I guess would move loop into here w/ syncs then change external loop to just when needing to cross boundaries
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return; // sounds like this way is generally better (more explicit that thread can stop)
            }
            int d2 = dExpo << 1;
            int k = (index) * d2;
            //if (k >= n) {
                //return; 
            //}
            //if (k < n) {
                //odata[k + d2 - 1] = idata[k + dExpo - 1] + idata[k + d2 - 1];
                // does this need a separate idata odata? I think no others operate on it this step
                data[k + d2 - 1] += data[k + dExpo - 1];
            //}
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
            // TODO not finished yet, doesn't work yet
            int blockSize = 128; // TODO optimize
            int dTarget = ilog2ceil(n);
            int pow2Size = 1 << dTarget;
            int nCap = pow2Size >> 1;
            dim3 fullBlocksPerGrid((nCap + blockSize - 1) / blockSize);
            // TODO I'm still not sure if getting fullBlocksPerGrid right 100%, might be overshooting
            int* dev_idata;
            int* dev_odata;

            // TODO note this pads to the whole next power of 2, was mentioned but can't recall if they said a way about that?
            // TODO I think want to rewrite into using shared memory way but that's extra credit so I think don't need to
            //   moving on for now but plan to revisit and do that
            //   also not sure if I've already done what part 5 is referring to? since I don't do modulo and such and decrease # of threads

            cudaMalloc((void**)&dev_idata, pow2Size * sizeof(int));
            //cudaMalloc((void**)&dev_odata, n * sizeof(int));
            
            cudaMemset(dev_idata + n, 0, sizeof(int) * (pow2Size - n));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice); 

            //cudaMemset(dev_idata, 0, sizeof(int));
            //cudaMemcpy(dev_idata + 1, idata, sizeof(int) * (n - 1), cudaMemcpyHostToDevice);

            // TODO make sure to deal with size not 2^x

            timer().startGpuTimer();

            // TODO
            
            // up-sweep
            int dExpo = 1; // = 2^(d)
            
            for (int d = 0; d < dTarget; ++d) {
                
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>> (nCap, dExpo, dev_idata);

                if (d < dTarget - 1) {
                    //fullBlocksPerGrid.x >>= 1;
                    //fullBlocksPerGrid = dim3((n / dExpo + blockSize - 1) / blockSize);
                    //fullBlocksPerGrid.x = (fullBlocksPerGrid.x - 1) / 2 + 1; // TODO reduce # of blocks accordingly
                    dExpo <<= 1;
                    nCap >>= 1;
                    fullBlocksPerGrid.x = ((nCap + blockSize - 1) / blockSize); // Not sure this is totally the best way to set this but does massively reduce runtime (e.g. ~6ms to ~2ms)
                    //std::swap(dev_idata, dev_odata);
                }
            }

            // down-sweep
            cudaMemset(dev_idata + (pow2Size - 1), 0, sizeof(int)); // TODO make sure that's right
            //fullBlocksPerGrid = dim3((pow2Size + blockSize - 1) / blockSize);
            // TODO make sure dExpo right
            for (int d = dTarget - 1; d >= 0; --d) {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(nCap, dExpo, dev_idata);
                // TODO make fullBlocksPerGrid right;
                //fullBlocksPerGrid.x = (dExpo + blockSize - 1) / blockSize; // TODO make sure set properly but works on super basic case
                if (d > 0) {
                    dExpo >>= 1;
                    nCap <<= 1;
                    fullBlocksPerGrid.x = ((nCap + blockSize - 1) / blockSize);
                }
            }


            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            //cudaFree(dev_odata);
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


            int blockSize = 128; // TODO optimize
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_idata;
            int* dev_odata;
            int* dev_boolArray;
            int* dev_indices;
            //int* dev_
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            int dTarget = ilog2ceil(n);
            int pow2Size = 1 << dTarget;
            int nCap = pow2Size >> 1;
            dim3 fullBlocksPerGridScan((nCap + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_boolArray, pow2Size * sizeof(int));
            cudaMalloc((void**)&dev_indices, pow2Size * sizeof(int));

            //cudaMemset(dev_idata, 0, sizeof(int));
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

                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (nCap, dExpo, dev_indices);

                if (d < dTarget - 1) {
                    dExpo <<= 1;
                    nCap >>= 1;
                    fullBlocksPerGridScan.x = ((nCap + blockSize - 1) / blockSize); // Not sure this is totally the best way to set this but does massively reduce runtime (e.g. ~6ms to ~2ms)
                }
            }

            // down-sweep
            cudaMemset(dev_indices + (pow2Size - 1), 0, sizeof(int));
            for (int d = dTarget - 1; d >= 0; --d) {
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (nCap, dExpo, dev_indices);
                if (d > 0) {
                    dExpo >>= 1;
                    nCap <<= 1;
                    fullBlocksPerGridScan.x = ((nCap + blockSize - 1) / blockSize);
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

            
            //return 10;
            return count + countStep;
        }
    }
}
