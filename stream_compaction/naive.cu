#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernScanStep(int n, int dExpo, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if (k >= dExpo) {
                odata[k] = idata[k - dExpo] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128; // TODO optimize
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int *dev_idata;
            int *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            
            cudaMemset(dev_idata, 0, sizeof(int));
            cudaMemcpy(dev_idata + 1, idata, sizeof(int) * (n - 1), cudaMemcpyHostToDevice); // TODO not sure best way to do this, but makes exclusive rather than inclusive
            cudaMemset(dev_odata, 0, sizeof(int) * n); // <- I think unnecessary?

            //this seems to perform worse than CPU but I assume that's expected when doing the naive approach since this obviously is inefficient, and I'm only testing on a small dataset anyway rn so the overhead of this outweighs any parallelism benefit
            timer().startGpuTimer();
            // TODO
            int dTarget = ilog2ceil(n);
            int dExpo = 1; // = 2^(d-1)
            for (int d = 1; d <= dTarget; ++d) {
                kernScanStep<<<fullBlocksPerGrid, blockSize>>>(n,dExpo,dev_odata,dev_idata);

                if (d < dTarget) {
                    dExpo *= 2;
                    std::swap(dev_idata, dev_odata);
                }
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
