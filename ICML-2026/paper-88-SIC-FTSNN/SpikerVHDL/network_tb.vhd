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
use ieee.std_logic_textio.all;
library std;
use std.textio.all;
library work;
use work.spiker_pkg.all;


entity network_tb is
end entity network_tb;

architecture behavior of network_tb is


    constant n_cycles : integer := 2;
    constant cycles_cnt_bitwidth : integer := 3;
    constant ready_filename : string := "ready.txt";
    constant sample_filename : string := "sample.txt";
    constant out_spikes_filename : string := "out_spikes.txt";
    constant neuron_dp_none_v_filename : string := "neuron_dp_none_v.txt";
    constant in_spikes_filename : string := "in_spikes.txt";

    component network is
        generic (
            n_cycles : integer := 2;
            cycles_cnt_bitwidth : integer := 3
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            sample_ready : in std_logic;
            ready : out std_logic;
            sample : out std_logic;
            in_spikes : in std_logic_vector(783 downto 0);
            out_spikes : out std_logic_vector(9 downto 0);
            neuron_dp_none_v : out signed(79 downto 0)
        );
    end component;


    signal clk : std_logic;
    signal rst_n : std_logic;
    signal start : std_logic;
    signal sample_ready : std_logic;
    signal ready : std_logic;
    signal sample : std_logic;
    signal in_spikes : std_logic_vector(783 downto 0);
    signal out_spikes : std_logic_vector(9 downto 0);
    signal neuron_dp_none_v : signed(79 downto 0);
    signal out_spikes_w_en : std_logic;
    signal neuron_dp_none_v_w_en : std_logic;
    signal in_spikes_rd_en : std_logic;

begin

    sample_ready <= '1';
    out_spikes_w_en <= sample;
    in_spikes_rd_en <= sample;
    neuron_dp_none_v_w_en <= sample;


    clk_gen : process
    begin

        clk <= '0';
        wait for 10 ns;
        clk <= '1';
        wait for 10 ns;

    end process clk_gen;

    rst_n_gen : process
    begin

        rst_n <= '1';
        wait for 15 ns;
        rst_n <= '0';
        wait for 10 ns;
        rst_n <= '1';

        wait;

    end process rst_n_gen;

    start_gen : process(clk )
    begin

        if clk'event and clk = '1' 
        then

            if ready = '1' 
            then

                start <= '1';

            else
                start <= '0';

            end if;

        end if;

    end process start_gen;

    out_spikes_save : process(clk, out_spikes_w_en )
        variable row : line;
        file out_spikes_file : text open write_mode is out_spikes_filename;
    begin

        if clk'event and clk = '1' 
        then

            if out_spikes_w_en = '1' 
            then

                write(row, std_logic_vector(out_spikes));
                writeline(out_spikes_file, row);

            end if;

        end if;

    end process out_spikes_save;

    neuron_dp_none_v_save : process(clk, neuron_dp_none_v_w_en )
        variable row : line;
        file neuron_dp_none_v_file : text open write_mode is neuron_dp_none_v_filename;
    begin

        if clk'event and clk = '1' 
        then

            if neuron_dp_none_v_w_en = '1' 
            then

                write(row, std_logic_vector(neuron_dp_none_v));
                writeline(neuron_dp_none_v_file, row);

            end if;

        end if;

    end process neuron_dp_none_v_save;

    in_spikes_load : process(clk, in_spikes_rd_en )
        variable row : line;
        variable read_var : std_logic_vector(783 downto 0);
        file in_spikes_file : text open read_mode is in_spikes_filename;
    begin

        if clk'event and clk = '1' 
        then

            if not endfile(in_spikes_file) 
            then

                if in_spikes_rd_en = '1' 
                then

                    readline(in_spikes_file, row);
                    read(row, read_var);
                    in_spikes <= read_var;

                end if;

            end if;

        end if;

    end process in_spikes_load;


    dut : network
        generic map(
            n_cycles => n_cycles,
            cycles_cnt_bitwidth => cycles_cnt_bitwidth
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start,
            sample_ready => sample_ready,
            ready => ready,
            sample => sample,
            in_spikes => in_spikes,
            out_spikes => out_spikes,
            neuron_dp_none_v => neuron_dp_none_v
        );


end architecture behavior;

