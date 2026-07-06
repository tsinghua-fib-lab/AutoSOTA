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


entity multi_input_128_exc_10_inh is
    generic (
        n_exc_inputs : integer := 128;
        n_inh_inputs : integer := 10;
        exc_cnt_bitwidth : integer := 7;
        inh_cnt_bitwidth : integer := 4
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        restart : in std_logic;
        start : in std_logic;
        exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
        inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
        neurons_ready : in std_logic;
        exc_cnt : out std_logic_vector(exc_cnt_bitwidth - 1 downto 0);
        inh_cnt : out std_logic_vector(inh_cnt_bitwidth - 1 downto 0);
        ready : out std_logic;
        neuron_restart : out std_logic;
        exc : out std_logic;
        inh : out std_logic;
        out_sample : out std_logic;
        exc_spike : out std_logic;
        inh_spike : out std_logic
    );
end entity multi_input_128_exc_10_inh;

architecture behavior of multi_input_128_exc_10_inh is


    component multi_input_dp_128_exc_10_inh is
        generic (
            n_exc_inputs : integer := 128;
            n_inh_inputs : integer := 10;
            exc_cnt_bitwidth : integer := 7;
            inh_cnt_bitwidth : integer := 4
        );
        port (
            clk : in std_logic;
            exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
            inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
            exc_sample : in std_logic;
            exc_rst_n : in std_logic;
            exc_cnt_en : in std_logic;
            exc_cnt_rst_n : in std_logic;
            inh_sample : in std_logic;
            inh_rst_n : in std_logic;
            inh_cnt_en : in std_logic;
            inh_cnt_rst_n : in std_logic;
            exc_yes : out std_logic;
            exc_spike : out std_logic;
            exc_stop : out std_logic;
            exc_cnt : out std_logic_vector(exc_cnt_bitwidth - 1 downto 0);
            inh_yes : out std_logic;
            inh_spike : out std_logic;
            inh_stop : out std_logic;
            inh_cnt : out std_logic_vector(inh_cnt_bitwidth - 1 downto 0)
        );
    end component;

    component multi_input_cu is
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            restart : in std_logic;
            start : in std_logic;
            neurons_ready : in std_logic;
            exc_yes : in std_logic;
            exc_stop : in std_logic;
            inh_yes : in std_logic;
            inh_stop : in std_logic;
            exc_cnt_en : out std_logic;
            exc_cnt_rst_n : out std_logic;
            inh_cnt_en : out std_logic;
            inh_cnt_rst_n : out std_logic;
            exc : out std_logic;
            inh : out std_logic;
            spike_sample : out std_logic;
            spike_rst_n : out std_logic;
            neuron_restart : out std_logic;
            ready : out std_logic
        );
    end component;


    signal spike_sample : std_logic;
    signal spike_rst_n : std_logic;
    signal exc_cnt_en : std_logic;
    signal exc_cnt_rst_n : std_logic;
    signal inh_cnt_en : std_logic;
    signal inh_cnt_rst_n : std_logic;
    signal exc_yes : std_logic;
    signal exc_stop : std_logic;
    signal inh_yes : std_logic;
    signal inh_stop : std_logic;

begin

    out_sample <= spike_sample;


    datapath : multi_input_dp_128_exc_10_inh
        generic map(
            n_exc_inputs => n_exc_inputs,
            n_inh_inputs => n_inh_inputs,
            exc_cnt_bitwidth => exc_cnt_bitwidth,
            inh_cnt_bitwidth => inh_cnt_bitwidth
        )
        port map(
            clk => clk,
            exc_spikes => exc_spikes,
            inh_spikes => inh_spikes,
            exc_sample => spike_sample,
            exc_rst_n => spike_rst_n,
            exc_cnt_en => exc_cnt_en,
            exc_cnt_rst_n => exc_cnt_rst_n,
            inh_sample => spike_sample,
            inh_rst_n => spike_rst_n,
            inh_cnt_en => inh_cnt_en,
            inh_cnt_rst_n => inh_cnt_rst_n,
            exc_yes => exc_yes,
            exc_spike => exc_spike,
            exc_stop => exc_stop,
            exc_cnt => exc_cnt,
            inh_yes => inh_yes,
            inh_spike => inh_spike,
            inh_stop => inh_stop,
            inh_cnt => inh_cnt
        );

    control_unit : multi_input_cu
        port map(
            clk => clk,
            rst_n => rst_n,
            restart => restart,
            start => start,
            neurons_ready => neurons_ready,
            exc_yes => exc_yes,
            exc_stop => exc_stop,
            inh_yes => inh_yes,
            inh_stop => inh_stop,
            exc_cnt_en => exc_cnt_en,
            exc_cnt_rst_n => exc_cnt_rst_n,
            inh_cnt_en => inh_cnt_en,
            inh_cnt_rst_n => inh_cnt_rst_n,
            exc => exc,
            inh => inh,
            spike_sample => spike_sample,
            spike_rst_n => spike_rst_n,
            neuron_restart => neuron_restart,
            ready => ready
        );


end architecture behavior;

