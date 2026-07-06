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
use ieee.numeric_std.all;


entity mux_4to1_signed is
    generic (
        bitwidth : integer := 8
    );
    port (
        mux_sel : in std_logic_vector(1 downto 0);
        in0 : in signed(bitwidth-1 downto 0);
        in1 : in signed(bitwidth-1 downto 0);
        in2 : in signed(bitwidth-1 downto 0);
        in3 : in signed(bitwidth-1 downto 0);
        mux_out : out signed(bitwidth-1 downto 0)
    );
end entity mux_4to1_signed;

architecture behavior of mux_4to1_signed is


begin

    selection : process(mux_sel, in0, in1, in2, in3 )
    begin

        case mux_sel is

            when "00" =>
                mux_out <= in0;


            when "01" =>
                mux_out <= in1;


            when "10" =>
                mux_out <= in2;


            when others =>
                mux_out <= in3;


        end case;

    end process selection;


end architecture behavior;

