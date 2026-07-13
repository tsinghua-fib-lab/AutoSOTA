#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace mcts {
    /**
     * A base context class for use in MctsEnv objects (see below).
     * 
     * This class is basically a wrapper around a map that is used as a context. Its main purpose is to provide a base 
     * class that can be used by the MctsEnv abstract class, and can be subclassed if a context more intricate than a 
     * map from strings is necessary.
     * 
     * Member variables:
     *      context: A mapping from string keys to abitrary void* pointers for arbitrary data store.
     */
    class MctsEnvContext {

        private:
            std::unordered_map<std::string, std::shared_ptr<void>> context;
            std::unordered_map<std::string, std::shared_ptr<const void>> context_const;

        public:
            /**
             * Mark destructor virtual in case class is inherited from
             */
            virtual ~MctsEnvContext() = default;

            /**
             * Gets a value from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (void*) stored in this context for the 'key'
             */
            virtual std::shared_ptr<void> get_value_raw(const std::string& key) const;

            /**
             * Gets a value from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (type T) stored in this context for the 'key'
            */
           template <typename T>
           T& get_value(const std::string& key) const {
                return *std::static_pointer_cast<T>(get_value_raw(key));
           }

            /**
             * Gets a value pointer from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value pointer (type T*) stored in this context for the 'key'
            */
           template <typename T>
           std::shared_ptr<T> get_value_ptr(const std::string& key) const {
                return std::static_pointer_cast<T>(get_value_raw(key));
           }

            /**
             * (Const version) Gets a value from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (void*) stored in this context for the 'key'
             */
            virtual std::shared_ptr<const void> get_value_raw_const(const std::string& key) const;

            /**
             * (Const version) Gets a value from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (type T) stored in this context for the 'key'
            */
           template <typename T>
           const T& get_value_const(const std::string& key) const {
                return *std::static_pointer_cast<const T>(get_value_raw_const(key));
           }

            /**
             * Gets a value pointer from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value pointer (type T*) stored in this context for the 'key'
            */
           template <typename T>
           std::shared_ptr<const T> get_value_ptr_const(const std::string& key) const {
                return std::static_pointer_cast<const T>(get_value_raw_const(key)) ;
           }

            /**
             * Puts a value in this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             *      val: A pointer to data correspdoning to 'key'
             */
            virtual void put_value_raw(const std::string& key, std::shared_ptr<void> val);

            /**
             * Puts a value in this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             *      val: A pointer to data correspdoning to 'key'
            */
           template <typename T>
           void put_value(const std::string& key, std::shared_ptr<T> val) {
                put_value_raw(key, std::static_pointer_cast<void>(val));
           }

            /**
             * (Const verson) Puts a value in this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             *      val: A pointer to data correspdoning to 'key'
             */
            virtual void put_value_raw_const(const std::string& key, std::shared_ptr<const void> val);

            /**
             * (Const version) Puts a value in this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             *      val: A pointer to data correspdoning to 'key'
            */
           template <typename T>
           void put_value_const(const std::string& key, std::shared_ptr<const T> val) {
                put_value_raw_const(key, std::static_pointer_cast<const void>(val));
           }

            /**
             * Erase a value from the context
            */
           virtual void erase(const std::string& key);

            /**
             * (Const version) Erase a value from the context
            */
           virtual void erase_const(const std::string& key);
    };
}