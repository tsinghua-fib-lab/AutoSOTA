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


entity multi_cycle_cu is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        start : in std_logic;
        all_ready : in std_logic;
        stop : in std_logic;
        cycles_cnt_en : out std_logic;
        cycles_cnt_rst_n : out std_logic;
        ready : out std_logic;
        restart : out std_logic;
        start_all : out std_logic
    );
end entity multi_cycle_cu;

architecture behavior of multi_cycle_cu is


    signal present_state : mc_states;
    signal next_state : mc_states;

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

    state_evaluation : process(present_state, start, all_ready, stop )
    begin

        case present_state is

            when reset =>
                next_state <= idle_wait;


            when idle_wait =>
                if all_ready = '1' 
                then

                    next_state <= idle;

                else
                    next_state <= idle_wait;

                end if;


            when idle =>
                if start = '1' 
                then

                    next_state <= init;

                else
                    next_state <= idle;

                end if;


            when init =>
                next_state <= update_wait;


            when update_wait =>
                if all_ready = '1' 
                then

                    if stop = '1' 
                    then

                        next_state <= idle;

                    else
                        next_state <= network_update;

                    end if;

                else
                    next_state <= update_wait;

                end if;


            when network_update =>
                next_state <= update_wait;


            when others =>
                next_state <= reset;


        end case;

    end process state_evaluation;

    output_evaluation : process(present_state )
    begin

        ready <= '0';
        start_all <= '0';
        cycles_cnt_en <= '0';
        cycles_cnt_rst_n <= '1';
        restart <= '0';

        case present_state is

            when reset =>
                cycles_cnt_rst_n <= '0';


            when idle_wait =>
                ready <= '0';


            when idle =>
                ready <= '1';
                cycles_cnt_rst_n <= '0';


            when init =>
                restart <= '1';


            when update_wait =>
                start_all <= '0';
                cycles_cnt_en <= '0';


            when network_update =>
                start_all <= '1';
                cycles_cnt_en <= '1';


            when others =>
                cycles_cnt_rst_n <= '0';


        end case;

    end process output_evaluation;


end architecture behavior;

