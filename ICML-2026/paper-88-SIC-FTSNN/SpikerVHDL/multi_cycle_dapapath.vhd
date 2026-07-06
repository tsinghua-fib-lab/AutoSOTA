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


entity multi_cycle_dapapath is
    generic (
        cycles_cnt_bitwidth : integer := 3;
        n_cycles : integer := 2
    );
    port (
        clk : in std_logic;
        cycles_cnt_en : in std_logic;
        cycles_cnt_rst_n : in std_logic;
        stop : out std_logic
    );
end entity multi_cycle_dapapath;

architecture behavior of multi_cycle_dapapath is


    component cnt is
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
    end component;

    component cmp_eq is
        generic (
            N : integer := 3
        );
        port (
            in0 : in std_logic_vector(N-1 downto 0);
            in1 : in std_logic_vector(N-1 downto 0);
            cmp_out : out std_logic
        );
    end component;


    signal cycles_cnt : std_logic_vector(cycles_cnt_bitwidth-1 downto 0);

begin

    cycles_counter : cnt
        generic map(
            N => cycles_cnt_bitwidth
        )
        port map(
            clk => clk,
            cnt_en => cycles_cnt_en,
            cnt_rst_n => cycles_cnt_rst_n,
            cnt_out => cycles_cnt
        );

    cycles_cmp : cmp_eq
        generic map(
            N => cycles_cnt_bitwidth
        )
        port map(
            in0 => cycles_cnt,
            in1 => std_logic_vector(to_unsigned(n_cycles + 1, cycles_cnt_bitwidth)),
            cmp_out => stop
        );


end architecture behavior;

