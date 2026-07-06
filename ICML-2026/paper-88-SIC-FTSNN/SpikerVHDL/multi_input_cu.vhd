---------------------------------------------------------------------------------
-- This is free and unencumbered software released into the public domain.
--
-- Anyone is free to copy, modify, publish, use, compile, sell, or
-- distribute this software, either in source code form or as a compiled
-- binary, for any purpose, commercial or non-commercial, and by any
-- means.
--
-- In jurisdictions that recognize copyright laws, the author or authors
-- of this software dedicate any and all copyright interest in the
-- software to the public domain. We make this dedication for the benefit
-- of the public at large and to the detriment of our heirs and
-- successors. We intend this dedication to be an overt act of
-- relinquishment in perpetuity of all present and future rights to this
-- software under copyright law.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
-- OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
-- ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
-- OTHER DEALINGS IN THE SOFTWARE.
--
-- For more information, please refer to <http://unlicense.org/>
---------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
library work;
use work.spiker_pkg.all;


entity multi_input_cu is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        restart : in std_logic;
        start : in std_logic;
        neurons_ready : in std_logic;
        exc_yes : in std_logic;
        exc_stop : in std_logic;
        inh_yes : in std_logic;
        inh_stop : in std_logic;
        exc_cnt_en : out std_logic;
        exc_cnt_rst_n : out std_logic;
        inh_cnt_en : out std_logic;
        inh_cnt_rst_n : out std_logic;
        exc : out std_logic;
        inh : out std_logic;
        spike_sample : out std_logic;
        spike_rst_n : out std_logic;
        neuron_restart : out std_logic;
        ready : out std_logic
    );
end entity multi_input_cu;

architecture behavior of multi_input_cu is


    signal present_state : mi_states;
    signal next_state : mi_states;

begin

    state_transition : process(clk, rst_n )
    begin

        if rst_n = '0' 
        then

            present_state <= reset;

        elsif(clk'event and clk = '1' )
        then

            present_state <= next_state;

        end if;

    end process state_transition;

    state_evaluation : process(present_state, restart, start, neurons_ready, exc_yes, exc_stop, inh_yes, inh_stop )
    begin

        case present_state is

            when reset =>
                next_state <= idle_wait;


            when idle_wait =>
                if neurons_ready = '1' 
                then

                    next_state <= idle;

                else
                    next_state <= idle_wait;

                end if;


            when idle =>
                if restart = '1' 
                then

                    next_state <= init_wait;

                else
                    if start = '1' 
                    then

                        next_state <= sample;

                    else
                        next_state <= idle;

                    end if;

                end if;


            when init_wait =>
                if neurons_ready = '1' 
                then

                    next_state <= init;

                else
                    next_state <= init_wait;

                end if;


            when init =>
                next_state <= idle;


            when sample =>
                if exc_yes = '1' 
                then

                    if inh_yes = '1' 
                    then

                        next_state <= exc_inh_wait;

                    else
                        next_state <= exc_wait;

                    end if;

                else
                    if inh_yes = '1' 
                    then

                        next_state <= inh_wait;

                    else
                        next_state <= idle;

                    end if;

                end if;


            when exc_inh_wait =>
                if neurons_ready = '1' 
                then

                    next_state <= exc_update_full;

                else
                    next_state <= exc_inh_wait;

                end if;


            when exc_update_full =>
                if exc_stop = '1' 
                then

                    next_state <= inh_wait_full;

                else
                    next_state <= exc_update_full;

                end if;


            when inh_wait_full =>
                if neurons_ready = '1' 
                then

                    next_state <= inh_update_full;

                else
                    next_state <= inh_wait_full;

                end if;


            when inh_update_full =>
                if inh_stop = '1' 
                then

                    next_state <= idle;

                else
                    next_state <= inh_update_full;

                end if;


            when exc_wait =>
                if neurons_ready = '1' 
                then

                    next_state <= exc_update;

                else
                    next_state <= exc_wait;

                end if;


            when inh_update =>
                if inh_stop = '1' 
                then

                    next_state <= idle;

                else
                    next_state <= inh_update;

                end if;


            when inh_wait =>
                if neurons_ready = '1' 
                then

                    next_state <= inh_update;

                else
                    next_state <= inh_wait;

                end if;


            when exc_update =>
                if exc_stop = '1' 
                then

                    next_state <= idle;

                else
                    next_state <= exc_update;

                end if;


            when others =>
                next_state <= reset;


        end case;

    end process state_evaluation;

    output_evaluation : process(present_state )
    begin

        exc_cnt_en <= '0';
        exc_cnt_rst_n <= '0';
        inh_cnt_en <= '0';
        inh_cnt_rst_n <= '0';
        exc <= '0';
        inh <= '0';
        spike_sample <= '0';
        spike_rst_n <= '1';
        neuron_restart <= '0';
        ready <= '0';

        case present_state is

            when reset =>
                exc_cnt_rst_n <= '0';
                inh_cnt_rst_n <= '0';
                spike_rst_n <= '0';


            when idle_wait =>
                ready <= '0';


            when idle =>
                ready <= '1';
                spike_rst_n <= '0';


            when init_wait =>
                ready <= '0';


            when init =>
                neuron_restart <= '1';


            when sample =>
                exc <= '1';
                inh <= '1';
                spike_sample <= '1';
                ready <= '0';


            when exc_inh_wait =>
                ready <= '0';


            when exc_update_full =>
                exc <= '1';
                exc_cnt_rst_n <= '1';
                exc_cnt_en <= '1';


            when inh_wait_full =>
                ready <= '0';


            when inh_update_full =>
                inh <= '1';
                inh_cnt_en <= '1';
                inh_cnt_rst_n <= '1';


            when exc_wait =>
                ready <= '0';


            when inh_update =>
                inh <= '1';
                inh_cnt_en <= '1';
                inh_cnt_rst_n <= '1';


            when inh_wait =>
                ready <= '0';


            when exc_update =>
                exc <= '1';
                exc_cnt_rst_n <= '1';
                exc_cnt_en <= '1';


            when others =>
                spike_rst_n <= '0';


        end case;

    end process output_evaluation;


end architecture behavior;

