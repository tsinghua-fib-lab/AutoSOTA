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


entity reg_sync_rst is
    generic (
        N : integer := 128
    );
    port (
        clk : in std_logic;
        en : in std_logic;
        rst_n : in std_logic;
        reg_in : in std_logic_vector(N-1 downto 0);
        reg_out : out std_logic_vector(N-1 downto 0)
    );
end entity reg_sync_rst;

architecture behavior of reg_sync_rst is


begin

    sample : process(clk, en, rst_n )
    begin

        if clk'event and clk = '1' 
        then

            if rst_n = '0' 
            then

                reg_out <= (others => '0');

            elsif(en = '1' )
            then

                reg_out <= reg_in;

            end if;

        end if;

    end process sample;


end architecture behavior;

