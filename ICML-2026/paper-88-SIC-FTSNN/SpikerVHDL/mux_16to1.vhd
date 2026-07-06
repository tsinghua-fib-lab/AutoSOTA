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


entity mux_16to1 is
    port (
        mux_sel : in std_logic_vector(3 downto 0);
        in0 : in std_logic;
        in1 : in std_logic;
        in2 : in std_logic;
        in3 : in std_logic;
        in4 : in std_logic;
        in5 : in std_logic;
        in6 : in std_logic;
        in7 : in std_logic;
        in8 : in std_logic;
        in9 : in std_logic;
        in10 : in std_logic;
        in11 : in std_logic;
        in12 : in std_logic;
        in13 : in std_logic;
        in14 : in std_logic;
        in15 : in std_logic;
        mux_out : out std_logic
    );
end entity mux_16to1;

architecture behavior of mux_16to1 is


begin

    selection : process(mux_sel, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15 )
    begin

        case mux_sel is

            when "0000" =>
                mux_out <= in0;


            when "0001" =>
                mux_out <= in1;


            when "0010" =>
                mux_out <= in2;


            when "0011" =>
                mux_out <= in3;


            when "0100" =>
                mux_out <= in4;


            when "0101" =>
                mux_out <= in5;


            when "0110" =>
                mux_out <= in6;


            when "0111" =>
                mux_out <= in7;


            when "1000" =>
                mux_out <= in8;


            when "1001" =>
                mux_out <= in9;


            when "1010" =>
                mux_out <= in10;


            when "1011" =>
                mux_out <= in11;


            when "1100" =>
                mux_out <= in12;


            when "1101" =>
                mux_out <= in13;


            when "1110" =>
                mux_out <= in14;


            when others =>
                mux_out <= in15;


        end case;

    end process selection;


end architecture behavior;

