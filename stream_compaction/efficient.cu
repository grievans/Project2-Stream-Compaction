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
            int d2 = dExpo * 2;
            int k = threadIdx.x + (blockIdx.x * blockDim.x) * d2;
            if (k >= n) {
                return;
            }
            //if (k < n) {
                //odata[k + d2 - 1] = idata[k + dExpo - 1] + idata[k + d2 - 1];
                // does this need a separate idata odata? I think no others operate on it this step
                data[k + d2 - 1] += data[k + dExpo - 1];
            //}
        }
        __global__ void kernDownSweep(int n, int dExpo, int* data) {
            int d2 = dExpo * 2;
            int k = threadIdx.x + (blockIdx.x * blockDim.x) * d2;
            if (k >= n) {
                return;
            }
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
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            //cudaMalloc((void**)&dev_odata, n * sizeof(int));
            
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice); 
            //cudaMemset(dev_odata, 0, sizeof(int) * n);

            //cudaMemset(dev_idata, 0, sizeof(int));
            //cudaMemcpy(dev_idata + 1, idata, sizeof(int) * (n - 1), cudaMemcpyHostToDevice);

            // TODO make sure to deal with size not 2^x

            timer().startGpuTimer();

            // TODO
            int dTarget = ilog2ceil(n); // TODO should these be out of timer?
            // up-sweep
            int dExpo = 1; // = 2^(d)
            for (int d = 0; d < dTarget; ++d) {
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>> (n, dExpo, dev_idata);

                if (d < dTarget) {
                    //fullBlocksPerGrid.x = (fullBlocksPerGrid.x - 1) / 2 + 1; // TODO reduce # of blocks accordingly
                    fullBlocksPerGrid = dim3((n / dExpo + blockSize - 1) / blockSize);
                    dExpo *= 2;
                    //std::swap(dev_idata, dev_odata);
                }
            }

            // down-sweep
            cudaMemset(dev_idata + (n - 1), 0, sizeof(int) * n); // TODO make sure that's right
            fullBlocksPerGrid = dim3((n + blockSize - 1) / blockSize);
            // TODO make sure dExpo right
            for (int d = dTarget - 1; d >= 0; --d) {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n, dExpo, dev_idata);
                // TODO make fullBlocksPerGrid right;
                dExpo /= 2;
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
