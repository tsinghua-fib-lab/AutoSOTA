#include "mcts_env_context.h"

using namespace std;

namespace mcts {
    /**
     * Implementation of 'get_value_for_key'
     * 
     * Just try to access the private context map. If there is an error in the access, we want it to be thrown anyway.
     */
    shared_ptr<void> MctsEnvContext::get_value_raw(const string& key) const {
        return context.at(key);
    }
    /**
     * (Const version) Implementation of 'get_value_for_key'
     * 
     * Just try to access the private context map. If there is an error in the access, we want it to be thrown anyway.
     */
    shared_ptr<const void> MctsEnvContext::get_value_raw_const(const string& key) const {
        return context_const.at(key);
    }

    /**
     * Implementation of 'put_value'
     */
    void MctsEnvContext::put_value_raw(const string& key, shared_ptr<void> val) {
        context[key] = val;
    }

    /**
     * (Const version) Implementation of 'put_value' for pointers to const types
     */
    void MctsEnvContext::put_value_raw_const(const string& key, shared_ptr<const void> val) {
        context_const[key] = val;
    }

    /**
     * Remove object from context
    */
   void MctsEnvContext::erase(const string& key) {
        context.erase(key);
   }

    /**
     * (Const version) Remove object from context
    */
   void MctsEnvContext::erase_const(const string& key) {
        context_const.erase(key);
   }
}