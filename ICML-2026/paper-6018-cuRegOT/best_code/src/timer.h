#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

// CUDA headers
#include <cuda_runtime.h>

// A simple class for recording time durations
class Timer
{
private:
    // https://stackoverflow.com/a/34781413
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    using TimePoint = std::chrono::time_point<Clock, Duration>;

    const bool m_enable;
    TimePoint m_start;
    std::unordered_map<std::string, double> m_durations;

public:
    // Set enable = false when we want to turn off timer
    Timer(bool enable = true):
        m_enable(enable)
    {}

    // Clear time points
    void clear()
    {
        m_durations.clear();
    }

    // Record starting time
    void tic()
    {
        if (m_enable)
        {
            m_start = Clock::now();
        }
    }

    // Record ending time and compute duration
    // Save {name: duration} to a hash map, and return the duration
    double toc(std::string name = "")
    {
        if (m_enable)
        {
            cudaDeviceSynchronize();
            TimePoint now = Clock::now();
            double duration = (now - m_start).count();
            m_durations[name] = duration;
            m_start = now;
            return duration;
        }
        return 0.0;
    }

    // Return the duration using key name
    // If the key does not exist, then 0 will be returned
    // (by adding {name: 0} to the hash map)
    double operator[](const std::string& name)
    {
        if (m_enable)
        {
            return m_durations[name];
        }
        return 0.0;
    }
};
