#pragma once

#include "helper.h"
#include "mcts_env.h"
#include "mcts_types.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <vector>


namespace mcts {
    // Forward declare
    class MctsEnv;

    /**
     * Args object so that params can be set in a more named args way
     * 
     * Member variables (that are different to RandManager/MctsManager):
     *      num_transposition_table_mutexes:
     *          Specifies the size of 'dmap_mutexes' in MctsManager
     *      seed:
     *          An integer seed to use for random number generation. Default of zero uses a 'random device' to generate 
     *          a seed 
     */
    struct MctsManagerArgs {
        static const int max_depth_default = std::numeric_limits<int>::max();
        // static const HeuristicFnPtr heuristic_fn_default = helper::zero_heuristic_fn;
        // static const PriorFnPtr prior_fn_default = nullptr;
        static const bool mcts_mode_default = true;
        static const bool is_two_player_game_default = false;
        static const bool use_transposition_table_default = false;
        static const int num_transposition_table_mutexes_default = 1;
        static const int seed_default = 0;
        
        std::shared_ptr<MctsEnv> mcts_env;
        int max_depth;
        HeuristicFnPtr heuristic_fn;
        PriorFnPtr prior_fn;

        bool mcts_mode;
        bool is_two_player_game;
        bool use_transposition_table;

        int num_transposition_table_mutexes;

        int seed;

        MctsManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            mcts_env(mcts_env),
            max_depth(max_depth_default),
            heuristic_fn(helper::zero_heuristic_fn),
            prior_fn(nullptr),
            mcts_mode(mcts_mode_default),
            is_two_player_game(is_two_player_game_default),
            use_transposition_table(use_transposition_table_default),
            num_transposition_table_mutexes(num_transposition_table_mutexes_default),
            seed(seed_default) {}

        virtual ~MctsManagerArgs() = default;
    };

    /**
     * Rand Manager. A manager for random number generation.
     * 
     * This class manages any random number generation needed, and is used as a base class for algorithm managers.
     * 
     * Member variables:
     *      rng_lock:
     *          A mutex to protect random number generation function calls.
     *      rd:
     *          A 'random_device' which is the computers source of (psuedo) random numbers
     *      int_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      real_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      int_distr: 
     *          A random number generator for integer numbers in the range [0,RAND_MAX)
     *      real_distr: 
     *          A random number generator for real numbers in the range [0,1)
     */
    class RandManager { 
        protected:
            std::mutex rng_lock;
            std::random_device rd;
            std::mt19937 int_gen;
            std::mt19937 real_gen;
            std::uniform_int_distribution<int> int_distr;
            std::uniform_real_distribution<double> real_distr;

            void init_random_seed() {
                int_gen = std::mt19937(rd());
                real_gen = std::mt19937(rd());
            }
        
        public:
            RandManager(const int seed=MctsManagerArgs::seed_default) :
                rng_lock(),
                int_gen(seed),
                real_gen(seed),
                int_distr(0,RAND_MAX),
                real_distr(0.0,1.0) 
            {
                if (seed == 0) init_random_seed();
            }

            /**
             * Returns a uniform random integer in the range [min_included, max_excluded).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual int get_rand_int(int min_included, int max_excluded) {
                std::lock_guard<std::mutex> lg(rng_lock);
                int len = int_distr(int_gen) % (max_excluded - min_included);
                return min_included + len;
            };
            
            /**
             * Returns a uniform random number in the range [0,1).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual double get_rand_uniform() {
                std::lock_guard<std::mutex> lg(rng_lock);
                return real_distr(real_gen);
            };
    };
    
    /**
     * MctsManager is an object used to manage all the things that need to be 'global' space within Mcts.
     * 
     * Primarily a Mcts manager stores all of the options that Mcts can be run with (see options section below).
     * 
     * As part of managing the 'global' space, the manager is responsible for storing the transposition table, 
     * although interaction with them is defined in mcts_decision_node.cpp and mcts_chance_node.cpp. Additionally it 
     * stores the pointers to the heuristic and prior function pointers that can be used by Mcts algorithms. 
     * 
     * Additionally, the Mcts manager is used to wrap any random number generation required, to provide a simple 
     * interface to node classes, without having to construct a random number generator in every node.
     * 
     * Member variables (environment):
     *      mcts_env:
     *          A MctsEnv object that provides the dynamics of the environment to plan in
     *      max_depth:
     *          The maximum depth that we want to allow our Mcts to search to.
     *      heuristic_fn:
     *          A pointer to the heuristic function to use. Defaults to return a constant zero value.
     *      prior_fn:
     *          A pointer to the prior function, that returns a map representing a policy. Defaults to nullptr to 
     *          indicate no prior. Prior may be able to be unormalised depending on the algorithm being used.
     * Member variables (options):
     *      mcts_mode:
     *          If mcts_mode is true, then only one node is added per trial (and initialised using the heuristic 
     *          function). If mcts_mode is false, then trials are run to completion (until max depth or a sink state is 
     *          reached).
     *      use_transposition_table:
     *          Specifies if a transposition table is to be used. Nodes are stored in a table upon creation, keyed by 
     *          (depth, Observation) tuples. When creating a new node, we first look if it exists in the table 
     *          already, and if it does we return that instead. This requires State and Action objects to have 
     *          std::hash and std::equal_to definitions. NOTE: should only use transposition_table if the 
     *          (depth,Observation) tuples have a one to one correspondance with decision nodes, otherwise this may 
     *          cause bugs.
     *      is_two_player_game:
     *          If we are planning for a two player game, rather than a reward maximisation environment
     * Member variables (transposition table):
     *      dmap:
     *          A transposition table for decision nodes. Note that a transposition table for chance nodes is 
     *          unnecessary, as an chance nodes that are transpositions, will be children of decision nodes that are 
     *          transpositions.
     *      dmap_mutexes:
     *          A vector of mutexes to use for protection around the dmap. Accessing 'dmap[dnode_id]', should be 
     *          protected by the 'dmap_mutexes[hash(dnode_id) % dmap_mutexes.size()]'.
     */
    class MctsManager : public RandManager {
        public:
            std::shared_ptr<MctsEnv> mcts_env;
            int max_depth;
            HeuristicFnPtr heuristic_fn;
            PriorFnPtr prior_fn;

            bool mcts_mode;
            bool use_transposition_table;
            bool is_two_player_game;

            DNodeTable dmap;
            std::vector<std::mutex> dmap_mutexes;

            /**
             * Constructor. Initialises values directly other than random number generation.
             * 
             * Seed is used to set the seed for cstdlib's rand(), and for the uniform random number generator. If the 
             * seed is set to 0, then we use a std::random_device object to generate a random seed.
             */    
            MctsManager(const MctsManagerArgs& args) : 
                RandManager(args.seed),
                mcts_env(args.mcts_env),
                max_depth(args.max_depth),
                heuristic_fn(args.heuristic_fn),
                prior_fn(args.prior_fn),
                mcts_mode(args.mcts_mode), 
                use_transposition_table(args.use_transposition_table), 
                is_two_player_game(args.is_two_player_game),
                dmap(),
                dmap_mutexes(args.num_transposition_table_mutexes)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~MctsManager() = default;
    };
}