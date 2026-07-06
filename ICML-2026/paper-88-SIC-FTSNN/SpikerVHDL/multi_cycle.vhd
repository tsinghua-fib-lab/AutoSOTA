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


entity multi_cycle is
    generic (
        cycles_cnt_bitwidth : integer := 3;
        n_cycles : integer := 2
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        start : in std_logic;
        all_ready : in std_logic;
        ready : out std_logic;
        restart : out std_logic;
        start_all : out std_logic
    );
end entity multi_cycle;

architecture behavior of multi_cycle is


    component multi_cycle_dapapath is
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
    end component;

    component multi_cycle_cu is
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
    end component;


    signal cycles_cnt_en : std_logic;
    signal cycles_cnt_rst_n : std_logic;
    signal stop : std_logic;

begin

    datapath : multi_cycle_dapapath
        generic map(
            cycles_cnt_bitwidth => cycles_cnt_bitwidth,
            n_cycles => n_cycles
        )
        port map(
            clk => clk,
            cycles_cnt_en => cycles_cnt_en,
            cycles_cnt_rst_n => cycles_cnt_rst_n,
            stop => stop
        );

    control_unit : multi_cycle_cu
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start,
            all_ready => all_ready,
            stop => stop,
            cycles_cnt_en => cycles_cnt_en,
            cycles_cnt_rst_n => cycles_cnt_rst_n,
            ready => ready,
            restart => restart,
            start_all => start_all
        );


end architecture behavior;

