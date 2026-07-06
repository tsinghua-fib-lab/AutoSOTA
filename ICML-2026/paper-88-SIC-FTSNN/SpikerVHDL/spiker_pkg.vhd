package spiker_pkg is

    type neuron_states is ( 
        reset,
        idle,
        init,
        excite,
        inhibit,
        fire,
        leak

    );

    type mi_states is ( 
        reset,
        idle_wait,
        idle,
        init_wait,
        init,
        sample,
        exc_inh_wait,
        exc_update_full,
        inh_wait_full,
        inh_update_full,
        exc_wait,
        inh_update,
        inh_wait,
        exc_update

    );

    type mc_states is ( 
        reset,
        idle_wait,
        idle,
        init,
        update_wait,
        network_update

    );

end package spiker_pkg;
