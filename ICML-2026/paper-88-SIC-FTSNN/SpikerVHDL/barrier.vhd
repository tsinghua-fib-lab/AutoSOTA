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


entity barrier is
    generic (
        N : integer := 10
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        restart : in std_logic;
        out_sample : in std_logic;
        reg_in : in std_logic_vector(N-1 downto 0);
        ready : out std_logic;
        reg_out : out std_logic_vector(N-1 downto 0)
    );
end entity barrier;

architecture behavior of barrier is


    component barrier_cu is
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            restart : in std_logic;
            out_sample : in std_logic;
            barrier_rst_n : out std_logic;
            barrier_en : out std_logic;
            ready : out std_logic
        );
    end component;

    component reg_sync_rst is
        generic (
            N : integer := 10
        );
        port (
            clk : in std_logic;
            en : in std_logic;
            rst_n : in std_logic;
            reg_in : in std_logic_vector(N-1 downto 0);
            reg_out : out std_logic_vector(N-1 downto 0)
        );
    end component;


    signal present_state : mi_states;
    signal next_state : mi_states;
    signal barrier_rst_n : std_logic;
    signal barrier_en : std_logic;

begin

    control_unit : barrier_cu
        port map(
            clk => clk,
            rst_n => rst_n,
            restart => restart,
            out_sample => out_sample,
            barrier_rst_n => barrier_rst_n,
            barrier_en => barrier_en,
            ready => ready
        );

    datapath : reg_sync_rst
        generic map(
            N => N
        )
        port map(
            clk => clk,
            en => barrier_en,
            rst_n => barrier_rst_n,
            reg_in => reg_in,
            reg_out => reg_out
        );


end architecture behavior;

