#include <cstdio>
#include "cpu.h"

#include "common.h"

#include <vector>

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            if (n > 0) {
                odata[0] = 0;
                for (int i = 1; i < n; ++i) {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            //return -1;
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n == 0) {
                return 0;
            }
            std::vector<int> b(n);
            std::vector<int> scanOut(n);

            //int* b = new int[n];
            //int* scanOut = new int[n];
            // TODO should I use something like this instead?: std::vector<bool> b(n);
            //  probably fine either way; I'm swapping to std::vector since that's my preference in my own code writing

            timer().startCpuTimer();
            // TODO
            
            // map
            for (int i = 0; i < n; ++i) {
                b[i] = idata[i] != 0 ? 1 : 0;
            }

            // scan
            /*scan(n, scanOut.data(), b.data());*/ // <- can't just invoke since conflict with timer, so just copying over
            scanOut[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanOut[i] = scanOut[i - 1] + b[i - 1];
            }

            // scatter
            for (int i = 0; i < n; ++i) {
                if (b[i]) { // or could use (b[i] == 1) but here should be fine
                    odata[scanOut[i]] = idata[i];
                }
            }

            int count = scanOut[n - 1];
            timer().endCpuTimer();
            //free(b);
            //free(scanOut);
            return count;
        }
    }
}
