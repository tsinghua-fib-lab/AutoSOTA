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


entity cnt is
    generic (
        N : integer := 8;
        rst_value : integer := 0
    );
    port (
        clk : in std_logic;
        cnt_en : in std_logic;
        cnt_rst_n : in std_logic;
        cnt_out : out std_logic_vector(N-1 downto 0)
    );
end entity cnt;

architecture behavior of cnt is


begin

    count : process(clk )
        variable cnt_var : integer := 0;
    begin

        if clk'event and clk = '1' 
        then

            if cnt_rst_n = '0' 
            then

                cnt_var := rst_value;

            else
                if cnt_en = '1' 
                then

                    cnt_var := cnt_var + 1;

                end if;

            end if;

        end if;

        cnt_out <= std_logic_vector(to_unsigned(cnt_var, N));

    end process count;


end architecture behavior;

