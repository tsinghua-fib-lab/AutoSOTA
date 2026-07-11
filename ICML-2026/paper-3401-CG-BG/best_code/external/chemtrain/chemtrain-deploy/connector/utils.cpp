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

#include "utils.h"

#include <functional>
#include <string>
#include <iostream>

namespace jcn {

    Logger::Logger() {
        // Reads the loglevel once

        char* loglevel_env = std::getenv("JCN_LOGLEVEL");
        if (loglevel_env == nullptr) {
            std::cout << "[JCN] Default log level set to "
            << levelToString(static_cast<LogLevel>(loglevel))
            << ". Set the JCN_LOGLEVEL environment variable to set a different "
            << "log level." << std::endl;
            return;
        };

        try {
            std::string val(loglevel_env);
            loglevel = std::stoi(val);

            return;
        } catch (const std::exception& e) {
            std::cerr << "[JCN] Invalid JCN_LOGLEVEL value: '" << loglevel_env 
                      << "'. Expected an integer. Error: " << e.what() << std::endl;
        }

        throw std::runtime_error("Invalid JCN_LOGLEVEL environment variable");

    }

    // Logs a message with a given log level
    void Logger::log(LogLevel level, const std::string& message)
    {

        // Loglevel not big enough
        if (!log(level)) return;

        std::cout << "[JCN:" << levelToString(level) << "] " << message << std::endl;

    }

    // Logs a message with a given log level
    bool Logger::log(LogLevel level)
    {

        return level <= static_cast<int>(loglevel);

    }

    std::string Logger::levelToString(LogLevel level) {
        switch (level) {
            case DEBUG:
                return "DEBUG";
            case INFO:
                return "INFO";
            case WARNING:
                return "WARNING";
            case ERROR:
                return "ERROR";
            case CRITICAL:
                return "CRITICAL";
            default:
                return "DEBUG";
        }
    }


} // namespace jcn
