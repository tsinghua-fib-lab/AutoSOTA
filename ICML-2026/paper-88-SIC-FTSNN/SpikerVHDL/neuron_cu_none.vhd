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


entity neuron_cu_none is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        restart : in std_logic;
        exc : in std_logic;
        inh : in std_logic;
        exceed_v_th : in std_logic;
        update_sel : out std_logic_vector(1 downto 0);
        add_or_sub : out std_logic;
        v_en : out std_logic;
        v_rst_n : out std_logic;
        neuron_ready : out std_logic;
        out_spike : out std_logic
    );
end entity neuron_cu_none;

architecture behavior of neuron_cu_none is


    signal present_state : neuron_states;
    signal next_state : neuron_states;

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

    state_evaluation : process(present_state, restart, exc, inh, exceed_v_th )
    begin

        case present_state is

            when reset =>
                next_state <= idle;


            when idle =>
                if restart = '1' 
                then

                    next_state <= init;

                else
                    if exc = '1' 
                    then

                        if inh = '1' 
                        then

                            if exceed_v_th = '1' 
                            then

                                next_state <= fire;

                            else
                                next_state <= leak;

                            end if;

                        else
                            next_state <= excite;

                        end if;

                    else
                        if inh = '1' 
                        then

                            next_state <= inhibit;

                        else
                            next_state <= idle;

                        end if;

                    end if;

                end if;


            when init =>
                next_state <= idle;


            when fire =>
                if exc = '1' and inh = '1' 
                then

                    next_state <= leak;

                else
                    next_state <= idle;

                end if;


            when leak =>
                if exc = '1' and inh = '1' 
                then

                    next_state <= leak;

                else
                    next_state <= idle;

                end if;


            when excite =>
                if exc = '1' 
                then

                    next_state <= excite;

                else
                    next_state <= idle;

                end if;


            when inhibit =>
                if inh = '1' 
                then

                    next_state <= inhibit;

                else
                    next_state <= idle;

                end if;


            when others =>
                next_state <= reset;


        end case;

    end process state_evaluation;

    output_evaluation : process(present_state )
    begin

        update_sel <= "01";
        add_or_sub <= '0';
        v_en <= '0';
        v_rst_n <= '1';
        out_spike <= '0';
        neuron_ready <= '0';

        case present_state is

            when reset =>
                v_rst_n <= '0';


            when idle =>
                neuron_ready <= '1';


            when init =>
                v_rst_n <= '0';


            when excite =>
                update_sel <= "10";
                v_en <= '1';


            when inhibit =>
                update_sel <= "11";
                v_en <= '1';


            when leak =>
                update_sel <= "01";
                v_en <= '1';
                add_or_sub <= '1';


            when fire =>
                v_en <= '1';
                out_spike <= '1';


            when others =>
                v_rst_n <= '0';


        end case;

    end process output_evaluation;


end architecture behavior;

