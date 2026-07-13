#include "mcts_types.h"

#include "helper_templates.h"

#include <functional> 
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace mcts;




namespace mcts {
    /**
     * Implementation of (null) Observation 
     */
    size_t Observation::hash() const {
        throw runtime_error("Trying to use default implementation of Observation::hash");
    }
    
    bool Observation::equals_itfc(const Observation& other) const {
        throw runtime_error("Trying to use default implementation of Observation::equals_itfc");
    }
    
    string Observation::get_pretty_print_string() const {
        throw runtime_error("Trying to use default implementation of Observation::get_pretty_print_string");
    }

    /**
     * Implementation of (null) State 
     */
    size_t State::hash() const {
        throw runtime_error("Trying to use default implementation of State::hash");
    }
    
    bool State::equals_itfc(const Observation& other) const {
        throw runtime_error("Trying to use default implementation of State::equals_itfc");
    }
    
    string State::get_pretty_print_string() const {
        throw runtime_error("Trying to use default implementation of State::get_pretty_print_string");
    }

    /**
     * Implementation of (null) Action 
     */
    size_t Action::hash() const {
        throw runtime_error("Trying to use default implementation of Action::hash");
    }
    
    bool Action::equals_itfc(const Action& other) const {
        throw runtime_error("Trying to use default implementation of Action::equals_itfc");
    }
    
    string Action::get_pretty_print_string() const {
        throw runtime_error("Trying to use default implementation of Action::get_pretty_print_string");
    }

    /**
     * Implementation of virtual hash function for IntState
     */
    size_t IntState::hash() const {
        return std::hash<int>()(state);
    }

    /**
     * Implementation of virtual equals_itfc function for IntPairState
     */
    bool IntState::equals_itfc(const Observation& other) const {
        try {
            const IntState& oth = dynamic_cast<const IntState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of equals function for IntState
     */
    bool IntState::equals(const IntState& other) const {
        return state == other.state;
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    string IntState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << state << ")";
        return ss.str();
    }
    
    /**
     * Implementation of virtual hash function for IntPairState
     */
    size_t IntPairState::hash() const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash,state.first);
        return helper::hash_combine(cur_hash, state.second);
    }

    /**
     * Implementation of virtual equals_itfc function for IntPairState
     */
    bool IntPairState::equals_itfc(const Observation& other) const {
        try {
            const IntPairState& oth = dynamic_cast<const IntPairState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for IntPairState
     */
    bool IntPairState::equals(const IntPairState& other) const {
        return state.first == other.state.first && state.second == other.state.second;
    }

    /**
     * Implementation of virtual equals function for IntPairState
     */
    string IntPairState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << state.first << "," << state.second << ")";
        return ss.str();
    }
    
    /**
     * Implementation of virtual hash function for Int3TupleState
     */
    size_t Int3TupleState::hash() const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash,get<0>(state));
        cur_hash = helper::hash_combine(cur_hash,get<1>(state));
        return helper::hash_combine(cur_hash,get<2>(state));
    }

    /**
     * Implementation of virtual equals_itfc function for Int3TupleState
     */
    bool Int3TupleState::equals_itfc(const Observation& other) const {
        try {
            const Int3TupleState& oth = dynamic_cast<const Int3TupleState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for Int3TupleState
     */
    bool Int3TupleState::equals(const Int3TupleState& other) const {
        return (get<0>(state) == get<0>(other.state)
            && get<1>(state) == get<1>(other.state)
            && get<2>(state) == get<2>(other.state));
    }

    /**
     * Implementation of virtual equals function for Int3TupleState
     */
    string Int3TupleState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << get<0>(state) << "," << get<1>(state) << "," << get<2>(state) << ")";
        return ss.str();
    }

    /**
     * Implementation of virtual hash function for IntAction
     */
    size_t IntAction::hash() const {
        return std::hash<int>()(action);
    }

    /**
     * Implementation of virtual equals_itfc function for IntAction
     */
    bool IntAction::equals_itfc(const Action& other) const {
        try {
            const IntAction& oth = dynamic_cast<const IntAction&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    bool IntAction::equals(const IntAction& other) const {
        return action == other.action;
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    string IntAction::get_pretty_print_string() const {
        stringstream ss;
        ss << action;
        return ss.str();
    }

    /**
     * Implementation of virtual hash function for StringAction
     */
    size_t StringAction::hash() const {
        return std::hash<string>()(action);
    }

    /**
     * Implementation of virtual equals_itfc function for StringAction
     */
    bool StringAction::equals_itfc(const Action& other) const {
        try {
            const StringAction& oth = dynamic_cast<const StringAction&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for StringAction
     */
    bool StringAction::equals(const StringAction& other) const {
        return action == other.action;
    }

    /**
     * Implementation of virtual equals function for StringAction
     */
    string StringAction::get_pretty_print_string() const {
        stringstream ss;
        ss << action;
        return ss.str();
    }
}

/**
 * Implementation of hash, equal_to and << operator functions for State/Action/Observation. Using virtual functions 
 * so that subclasses only need to implement those sub functions
 */
namespace std {
    /**
     * Implementation of std::hash<Observation>, calling the virtual hash function.
     */
    size_t hash<Observation>::operator()(const Observation& observation) const {
        return observation.hash();
    }

    size_t hash<shared_ptr<const Observation>>::operator()(const shared_ptr<const Observation>& observation) const {
        return observation->hash();
    }

    /**
     * Implementation of std::equal_to<Observation>, calling the equals function.
     */
    bool operator==(const Observation& lhs, const Observation& rhs) {
        return lhs.equals_itfc(rhs);
    }

    bool operator==(const shared_ptr<const Observation>& lhs, const shared_ptr<const Observation>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<Observation>::operator()(const Observation& lhs, const Observation& rhs) const {
        return lhs.equals_itfc(rhs);
    }

    bool equal_to<shared_ptr<const Observation>>::operator()(
        const shared_ptr<const Observation>& lhs, const shared_ptr<const Observation>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }

    /**
     * Override output stream << operator for Observation, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const Observation& observation) {
        os << observation.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const Observation>& observation) {
        os << observation->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<State>, calling the virtual hash function.
     */
    size_t hash<State>::operator()(const State& state) const {
        return state.hash();
    }
    
    size_t hash<shared_ptr<const State>>::operator()(const shared_ptr<const State>& state) const {
        return state->hash();
    }
    /**
     * Implementation of std::equal_to<State>, calling the equals function.
     */
    bool operator==(const State& lhs, const State& rhs) {
        return lhs.equals_itfc(rhs);
    }

    bool operator==(const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<State>::operator()(const State& lhs, const State& rhs) const {
        return lhs.equals_itfc(rhs);
    }

    bool equal_to<shared_ptr<const State>>::operator()(
        const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }

    /**
     * Override output stream << operator for State, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const State& state) {
        os << state.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const State>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<Action>, calling the virtual hash function.
     */
    size_t hash<Action>::operator()(const Action& action) const {
        return action.hash();
    }

    size_t hash<shared_ptr<const Action>>::operator()(const shared_ptr<const Action>& action) const {
        return action->hash();
    }

    /**
     * Implementation of std::equal_to<Action>, calling the equals function.
     */
    bool operator==(const Action& lhs, const Action& rhs) {
        return lhs.equals_itfc(rhs);
    }

    bool operator==(const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<Action>::operator()(const Action& lhs, const Action& rhs) const {
        return lhs.equals_itfc(rhs);
    }

    bool equal_to<shared_ptr<const Action>>::operator()(
        const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }

    /**
     * Override output stream << operator for Action, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const Action& action) {
        os << action.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const Action>& action) {
        os << action->get_pretty_print_string();
        return os;
    }

    /**
     * Overrride output strea m<< operator for derived State and Action classes
     */
    ostream& operator<<(ostream& os, const IntState& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const IntState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const IntPairState& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const IntPairState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const Int3TupleState& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const Int3TupleState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const IntAction& action) {
        os << action.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const IntAction>& action) {
        os << action->get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const StringAction action) {
        os << action.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const StringAction>& action) {
        os << action->get_pretty_print_string();
        return os;
    }
}

/**
 * Implementation of << operator for common typedefs
 */
namespace std {

    ostream& operator<<(ostream& os, const ActionVector& vec) {
        os << helper::vector_pretty_print_string(vec);
        return os;
    }

    ostream& operator<<(ostream& os, const StringActionVector& vec) {
        os << helper::vector_pretty_print_string(vec);
        return os;
    }

    ostream& operator<<(ostream& os, const StateDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }

    ostream& operator<<(ostream& os, const ObservationDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }

    ostream& operator<<(ostream& os, const IntPairStateDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }

    ostream& operator<<(ostream& os, const Int3TupleStateDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }
}

/**
 * Implementation of hash, equals_to and output stream functions for transposition table types
 */
namespace std {
    /**
     * Implementation of std::hash<DNodeIdTuple>.
     */
    size_t hash<DNodeIdTuple>::operator()(const DNodeIdTuple& tpl) const {
        size_t hash_val = 0;
        hash_val = helper::hash_combine(hash_val, get<0>(tpl));
        hash_val = helper::hash_combine(hash_val, get<1>(tpl));
        return hash_val;
    }

    /**
     * Implementation of std::equal_to<DNodeIdTuple>.
     */
    bool equal_to<DNodeIdTuple>::operator()(const DNodeIdTuple& lhs, const DNodeIdTuple& rhs) const {
        return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs);
    }

    /**
     * Override output stream << operator for DNodeIdTuple.
     */
    ostream& operator<<(ostream& os, const DNodeIdTuple& tpl) {
        os << "DNodeId(" << get<0>(tpl) << "," << get<1>(tpl) << ")";
        return os;
    }

    /**
     * Override output stream << operator for DNodeTable.
     */
    ostream& operator<<(ostream& os, const DNodeTable& tbl) {
        unordered_map<DNodeIdTuple,shared_ptr<MctsDNode>> shared_tbl_for_print;
        for (auto pr : tbl) {
            shared_tbl_for_print.insert_or_assign(pr.first, shared_ptr<MctsDNode>(pr.second));
        }
        os << helper::unordered_map_pretty_print_string(shared_tbl_for_print);
        return os;
    }
}