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


entity add_sub is
    generic (
        N : integer := 8
    );
    port (
        in0 : in signed(N-1 downto 0);
        in1 : in signed(N-1 downto 0);
        add_or_sub : in std_logic;
        add_sub_out : out signed(N-1 downto 0)
    );
end entity add_sub;

architecture behavior of add_sub is


    constant sat_up : integer := 2**(N-1)-1;
    constant sat_down : integer := -2**(N-1);

    signal local_in0 : signed(N downto 0);
    signal local_in1 : signed(N downto 0);
    signal local_out : signed(N downto 0);

begin

    local_in0 <= in0(N-1) & in0;
    local_in1 <= in1(N-1) & in1;


    sat_add_sub : process(local_in0, local_in1, local_out, add_or_sub )
    begin

        if add_or_sub = '0' 
        then

            local_out <= local_in0 + local_in1;

        else
            local_out <= local_in0 - local_in1;

        end if;
        if local_out(N) /= local_out(N-1) 
        then

            if local_out(N) = '1' 
            then

                add_sub_out <= to_signed(sat_down, N);

            else
                add_sub_out <= to_signed(sat_up, N);

            end if;

        else
            add_sub_out <= local_out(N-1 downto 0);

        end if;

    end process sat_add_sub;


end architecture behavior;

