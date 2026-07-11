/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <string>
#include <iostream>


namespace jcn {

    // Enum to represent log levels
    enum LogLevel {
        DEBUG = 2,
        INFO = 1,
        WARNING = 0,
        ERROR = -1,
        CRITICAL = -2
    };

    class Logger {
    public:
        ~Logger() = default;

        // Singleton instance
        static Logger& getlogger() {
            static Logger instance;
            return instance;
        }

        // Logs a message with a given log level
        void log(LogLevel level, const std::string& message);

        // Checks if the log level is sufficiently high
        bool log(LogLevel level);

    private:
        Logger();

        int loglevel = 0;

        std::string levelToString(LogLevel level);

    };

} // namespace jcn



#endif //UTILS_H
