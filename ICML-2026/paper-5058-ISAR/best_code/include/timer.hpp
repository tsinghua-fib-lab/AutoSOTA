#ifndef TIMER_HPP
#define TIMER_HPP

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>

class ScopedTimer{
public:
    ScopedTimer(std::string name) : _name(name),_t_begin(std::chrono::steady_clock::now()){};
    ~ScopedTimer() {
        std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - _t_begin).count();
        auto& data = _timers[_name];
        data.ms += elapsed;
        data.count++;
        if (data.max_ms < elapsed)
            data.max_ms = elapsed;
    }

    struct TimerData{
        int64_t ms = 0;
        int count = 0;
        int64_t max_ms = 0;
    };

    static void print_timers() {
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::setw(40) << std::left << "Timer" << std::right << std::setw(9) << "Count" << "    " << std::setw(13) << "Total-Time" << "     " << std::setw(9) << "Max-Time" << std::endl;
        for(const auto& kvp : _timers) {
            std::cout << std::setw(40) << std::left << kvp.first << std::right << std::setw(9) << kvp.second.count << "    " << std::setw(13) << (std::to_string(kvp.second.ms) + " ms") << "     " << std::setw(9) << (std::to_string(kvp.second.max_ms) + " ms") << std::endl;
        }
        std::cout << std::string(80, '-') << "\n" << std::endl;
    }

    static void reset_timers() {
        _timers.clear();
    }

private:
    std::string _name;
    std::chrono::steady_clock::time_point _t_begin;
    static std::map<std::string, TimerData> _timers;
};

inline std::map<std::string, ScopedTimer::TimerData> ScopedTimer::_timers;

#endif //TIMER_HPP