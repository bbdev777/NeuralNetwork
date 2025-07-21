
#include <chrono>

namespace AppExecutionTimeCounter
{
    static auto st = std::chrono::high_resolution_clock::now();

    static void StartMeasurement()
    {
        st = std::chrono::high_resolution_clock::now();
    }

    static double EndMeasurement()
    {
        auto et = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedSeconds = et - st;
	    
        return elapsedSeconds.count();
    }
}