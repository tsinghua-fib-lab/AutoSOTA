#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * mcts_types.h
 * 
 * This file contains some base types used by Mcts, namely State, Action and Observation classes.
 * 
 * These implementations are just empty classes which serve as a supertype to be used in Mcts classes.
 * The State class inherits from the Observation class so that we can return a State as an Observation, as Observations 
 * can be the state in fully observable environments. 
 * 
 * State, Action and Observation objects need to be used as keys in unordered_map objects. Which means that they need 
 * to implement the relevent std::hash<State>, std::equal_to<State> and std::hash<Action>, std::equal_to<Action> and 
 * std::hash<Observation>, std::equal_to<Observation> classes.
 */

namespace mcts {
    // Forward declare Mcts env
    class MctsEnv;

    /**
     * A abstract base type to use for Observations.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In mcts_types.cpp, we 
     * implement the std::hash<Observation> and std::equal_to<Observation> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     * 
     * N.B. Implementations are provided, but are such that a direct instance of Observation is equivalent to a 
     * 'NullObservation'.
     */
    class Observation {
        public:
            virtual ~Observation() = default;
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * A abstract base type to use for States.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In mcts_types.cpp, we 
     * implement the std::hash<State> and std::equal_to<State> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     * 
     * N.B. Implementations are provided, but are such that a direct instance of State is equivalent to a 
     * 'NullState'.
     */
    class State : public Observation {
        public:
            virtual ~State() = default;
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };
    


    /**
     * A abstract base type to use for Actions.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In mcts_types.cpp, we 
     * implement the std::hash<Action> and std::equal_to<Action> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     * 
     * N.B. Implementations are provided, but are such that a direct instance of Action is equivalent to a 
     * 'NullAction'.
     */
    class Action {
        public:
            virtual ~Action() = default;
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Action& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * An implementaton of state containing a single integer state.
     */
    class IntState : public State {
        public:
            int state;

            IntState(int state) : state(state) {}
            virtual ~IntState() = default;
            virtual std::size_t hash() const;
            bool equals(const IntState& other) const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * An implementaton of state containing a pair of integers as the state.
     */
    class IntPairState : public State {
        public:
            std::pair<int,int> state;

            IntPairState(std::pair<int,int> pr) : state(pr) {}
            IntPairState(int first, int second) : state(std::make_pair(first,second)) {}
            virtual ~IntPairState() = default;
            virtual std::size_t hash() const;
            bool equals(const IntPairState& other) const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * An implementaton of state containing a 3 tuple of integers as the state.
     */
    class Int3TupleState : public State {
        public:
            std::tuple<int,int,int> state;

            Int3TupleState(std::tuple<int,int,int> tpl) : state(tpl) {}
            Int3TupleState(int first, int second, int third) : state(std::make_tuple(first,second,third)) {}
            virtual ~Int3TupleState() = default;
            virtual std::size_t hash() const;
            bool equals(const Int3TupleState& other) const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * An implementation of action containing a single int as an action
     */
    class IntAction : public Action {
        public:
            int action;

            IntAction(int action) : action(action) {}
            virtual ~IntAction() = default;
            virtual std::size_t hash() const;
            bool equals(const IntAction& other) const;
            virtual bool equals_itfc(const Action& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * An implementation of action containing a single string as an action
     */
    class StringAction : public Action {
        public:
            std::string action;

            StringAction(std::string action) : action(action) {}
            virtual ~StringAction() = default;
            virtual std::size_t hash() const;
            bool equals(const StringAction& other) const;
            virtual bool equals_itfc(const Action& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * Typedef for heuristic function pointers
     * First used in mcts_decision_node.h and mcts_chance_node.h
     * N.B. The & here is to get address as we want function pointers
     */  
    double _DummyHeuristicFn(std::shared_ptr<const State> s, std::shared_ptr<MctsEnv> env);
    typedef decltype(&_DummyHeuristicFn) HeuristicFnPtr;

    /**
     * Typedef for (action) prior function pointers
     * First used in mcts_decision_node.h and mcts_chance_node.h
     * N.B. The & here is to get address as we want function pointers
     */
    typedef std::unordered_map<std::shared_ptr<const Action>,double> ActionPrior;
    std::shared_ptr<ActionPrior> _DummyPriorFn(std::shared_ptr<const State> s, std::shared_ptr<MctsEnv> env);
    typedef decltype(&_DummyPriorFn) PriorFnPtr; 



    /**
     * Commonly want a list of actions, and the type is a bit verbose.
     */
    typedef std::vector<std::shared_ptr<const Action>> ActionVector;
    typedef std::vector<std::shared_ptr<const StringAction>> StringActionVector;
    typedef std::vector<std::shared_ptr<const IntAction>> IntActionVector;

    /**
     * Also commonly want a distribution over states/actions/observations, and the types are verbose.
     */
    typedef std::unordered_map<std::shared_ptr<const State>,double> StateDistr;
    typedef std::unordered_map<std::shared_ptr<const Action>,double> ActionDistr;
    typedef std::unordered_map<std::shared_ptr<const Observation>,double> ObservationDistr;
    typedef std::unordered_map<std::shared_ptr<const IntState>,double> IntStateDistr;
    typedef std::unordered_map<std::shared_ptr<const IntPairState>,double> IntPairStateDistr;
    typedef std::unordered_map<std::shared_ptr<const Int3TupleState>,double> Int3TupleStateDistr;



    // forward declare
    class MctsDNode;
    class MctsCNode;

    /**
     * Typedef for dnode id tuples, for readability.
     * First used transposition table implementation, in mcts_manager.h, mcts_decision_node.h and mcts_chance_node.h
     * 
     * The DNodeTable has a weak_ptr. If it were shared then MctsManager would have shared_ptr's to MctsDNode's, and 
     * MctsDNode's have a shared_ptr to the same MctsManager. The circular dependency leads to reference counting not 
     * working and hence memory leaks when using the transposition table otherwise.
     */
    typedef std::tuple<int,std::shared_ptr<const Observation>> DNodeIdTuple;
    typedef std::unordered_map<DNodeIdTuple,std::weak_ptr<MctsDNode>> DNodeTable;
}

/**
 * Forward declaring the hash, equality and output stream functions defined in mcts_types.cpp.
 * Needed so other files know to look at mcts_types.o to find implementations of these functions.
 */
namespace std {
    using namespace mcts;

    /**
     * Hash, equality class and output stream function definitions for Observation.
     */
    template <> 
    struct hash<Observation> {
        size_t operator()(const Observation&) const;
    };

    template <> 
    struct hash<shared_ptr<const Observation>> {
        size_t operator()(const shared_ptr<const Observation>&) const;
    };
    
    bool operator==(const Observation& lhs, const Observation& rhs);
    bool operator==(const shared_ptr<const mcts::Observation>& lhs, const shared_ptr<const mcts::Observation>& rhs);

    template <> 
    struct equal_to<Observation> {
        bool operator()(const Observation&, const Observation&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const Observation>> {
        bool operator()(const shared_ptr<const Observation>&, const shared_ptr<const Observation>&) const;
    };

    ostream& operator<<(ostream& os, const Observation& observation);
    ostream& operator<<(ostream& os, const shared_ptr<const Observation>& observation);

    /**
     * Hash, equality class and output stream function definitions for State.
     */
    template <> 
    struct hash<State> {
        size_t operator()(const State&) const;
    };

    template <> 
    struct hash<std::shared_ptr<const State>> {
        size_t operator()(const shared_ptr<const State>&) const;
    };
    
    bool operator==(const State& lhs, const State& rhs);
    bool operator==(const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs);

    template <> 
    struct equal_to<State> {
        bool operator()(const State&, const State&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const State>> {
        bool operator()(const shared_ptr<const State>&, const shared_ptr<const State>&) const;
    };

    ostream& operator<<(ostream& os, const State& state);
    ostream& operator<<(ostream& os, const shared_ptr<const State>& state);

    /**
     * Hash, equality class and output stream function definitions for Action.
     */
    template <> 
    struct hash<Action> {
        size_t operator()(const Action&) const;
    };

    template <> 
    struct hash<shared_ptr<const Action>> {
        size_t operator()(const shared_ptr<const Action>&) const;
    };
    
    bool operator==(const Action& lhs, const Action& rhs);
    bool operator==(const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs);

    template <> struct equal_to<Action> {
        bool operator()(const Action&, const Action&) const;
    };

    template <> struct equal_to<shared_ptr<const Action>> {
        bool operator()(const shared_ptr<const Action>&, const shared_ptr<const Action>&) const;
    };

    ostream& operator<<(ostream& os, const Action& action);
    ostream& operator<<(ostream& os, const shared_ptr<const Action>& Action);

    /**
     * Output stream overloeads for Observation, State and Action subclasses defined above
     */ 
    ostream& operator<<(ostream& os, const IntState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const IntState>& state);
    ostream& operator<<(ostream& os, const IntPairState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const IntPairState>& state);
    ostream& operator<<(ostream& os, const Int3TupleState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const Int3TupleState>& state);
    ostream& operator<<(ostream& os, const IntAction& action);
    ostream& operator<<(ostream& os, const shared_ptr<const IntAction>& action);
    ostream& operator<<(ostream& os, const StringAction& action);
    ostream& operator<<(ostream& os, const shared_ptr<const StringAction>& action);

    /**
     * Output stream overloeads for common vector and map typedefs
     */ 
    ostream& operator<<(ostream& os, const ActionVector& vec);
    ostream& operator<<(ostream& os, const StringActionVector& vec);
    ostream& operator<<(ostream& os, const StateDistr& distr);
    ostream& operator<<(ostream& os, const ObservationDistr& distr);
    ostream& operator<<(ostream& os, const IntPairStateDistr& distr);
    ostream& operator<<(ostream& os, const Int3TupleStateDistr& distr);

    /**
     * Hash, equality and stream functions for DNodeIdTuple
     */
    template <> 
    struct hash<DNodeIdTuple> {
        size_t operator()(const DNodeIdTuple&) const;
    };

    template <> 
    struct equal_to<DNodeIdTuple> {
        bool operator()(const DNodeIdTuple&, const DNodeIdTuple&) const;
    };

    ostream& operator<<(ostream& os, const DNodeIdTuple& tpl);

    /**
     * Output streams for transposition tables
     */
    ostream& operator<<(ostream& os, const DNodeTable& tbl);
}