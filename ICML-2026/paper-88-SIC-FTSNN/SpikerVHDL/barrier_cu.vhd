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


entity barrier_cu is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        restart : in std_logic;
        out_sample : in std_logic;
        barrier_rst_n : out std_logic;
        barrier_en : out std_logic;
        ready : out std_logic
    );
end entity barrier_cu;

architecture behavior of barrier_cu is


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

    state_evaluation : process(present_state, restart, out_sample )
    begin

        case present_state is

            when reset =>
                next_state <= idle;


            when idle =>
                if restart = '1' 
                then

                    next_state <= init;

                else
                    if out_sample = '1' 
                    then

                        next_state <= sample;

                    else
                        next_state <= idle;

                    end if;

                end if;


            when init =>
                next_state <= idle;


            when sample =>
                next_state <= idle;


            when others =>
                next_state <= idle;


        end case;

    end process state_evaluation;

    output_evaluation : process(present_state )
    begin

        ready <= '0';
        barrier_rst_n <= '1';
        barrier_en <= '0';

        case present_state is

            when reset =>
                barrier_rst_n <= '0';


            when idle =>
                ready <= '1';


            when init =>
                barrier_rst_n <= '0';


            when sample =>
                barrier_en <= '1';


            when others =>
                ready <= '1';


        end case;

    end process output_evaluation;


end architecture behavior;

