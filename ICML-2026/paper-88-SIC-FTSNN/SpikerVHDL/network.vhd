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
library work;
use work.spiker_pkg.all;


entity network is
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
end entity network;

architecture behavior of network is


    component multi_cycle is
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
    end component;

    component layer_1024_neurons_784_inputs is
        generic (
            n_exc_inputs : integer := 784;
            n_inh_inputs : integer := 1024;
            exc_cnt_bitwidth : integer := 10;
            inh_cnt_bitwidth : integer := 10;
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            restart : in std_logic;
            exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
            inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
            ready : out std_logic;
            out_spikes : out std_logic_vector(1023 downto 0)
        );
    end component;

    component layer_512_neurons_1024_inputs is
        generic (
            n_exc_inputs : integer := 1024;
            n_inh_inputs : integer := 512;
            exc_cnt_bitwidth : integer := 10;
            inh_cnt_bitwidth : integer := 9;
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            restart : in std_logic;
            exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
            inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
            ready : out std_logic;
            out_spikes : out std_logic_vector(511 downto 0)
        );
    end component;

    component layer_128_neurons_512_inputs is
        generic (
            n_exc_inputs : integer := 512;
            n_inh_inputs : integer := 128;
            exc_cnt_bitwidth : integer := 9;
            inh_cnt_bitwidth : integer := 7;
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            restart : in std_logic;
            exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
            inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
            ready : out std_logic;
            out_spikes : out std_logic_vector(127 downto 0)
        );
    end component;

    component layer_10_neurons_128_inputs is
        generic (
            n_exc_inputs : integer := 128;
            n_inh_inputs : integer := 10;
            exc_cnt_bitwidth : integer := 7;
            inh_cnt_bitwidth : integer := 4;
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            restart : in std_logic;
            exc_spikes : in std_logic_vector(n_exc_inputs-1 downto 0);
            inh_spikes : in std_logic_vector(n_inh_inputs-1 downto 0);
            ready : out std_logic;
            out_spikes : out std_logic_vector(9 downto 0);
            neuron_dp_none_v : out signed(79 downto 0)
        );
    end component;


    signal start_all : std_logic;
    signal all_ready : std_logic;
    signal restart : std_logic;
    signal layer_0_ready : std_logic;
    signal layer_0_feedback : std_logic_vector(1023 downto 0);
    signal layer_1_ready : std_logic;
    signal layer_1_feedback : std_logic_vector(511 downto 0);
    signal exc_spikes_0_to_1 : std_logic_vector(1023 downto 0);
    signal layer_2_ready : std_logic;
    signal layer_2_feedback : std_logic_vector(127 downto 0);
    signal exc_spikes_1_to_2 : std_logic_vector(511 downto 0);
    signal layer_3_ready : std_logic;
    signal layer_3_feedback : std_logic_vector(9 downto 0);
    signal exc_spikes_2_to_3 : std_logic_vector(127 downto 0);

begin

    sample <= start_all;
    all_ready <= sample_ready and layer_0_ready and layer_1_ready and layer_2_ready and layer_3_ready ;
    out_spikes <= layer_3_feedback;
    exc_spikes_0_to_1<= layer_0_feedback;
    exc_spikes_1_to_2<= layer_1_feedback;
    exc_spikes_2_to_3<= layer_2_feedback;


    multi_cycle_control : multi_cycle
        generic map(
            cycles_cnt_bitwidth => cycles_cnt_bitwidth,
            n_cycles => n_cycles
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start,
            all_ready => all_ready,
            ready => ready,
            restart => restart,
            start_all => start_all
        );

    layer_0 : layer_1024_neurons_784_inputs
        generic map(
            n_exc_inputs => 784,
            n_inh_inputs => 1024,
            exc_cnt_bitwidth => 10,
            inh_cnt_bitwidth => 10,
            neuron_bit_width => 8,
            shift => 4
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start_all,
            restart => restart,
            exc_spikes => in_spikes,
            inh_spikes => layer_0_feedback,
            ready => layer_0_ready,
            out_spikes => layer_0_feedback
        );

    layer_1 : layer_512_neurons_1024_inputs
        generic map(
            n_exc_inputs => 1024,
            n_inh_inputs => 512,
            exc_cnt_bitwidth => 10,
            inh_cnt_bitwidth => 9,
            neuron_bit_width => 8,
            shift => 4
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start_all,
            restart => restart,
            exc_spikes => exc_spikes_0_to_1,
            inh_spikes => layer_1_feedback,
            ready => layer_1_ready,
            out_spikes => layer_1_feedback
        );

    layer_2 : layer_128_neurons_512_inputs
        generic map(
            n_exc_inputs => 512,
            n_inh_inputs => 128,
            exc_cnt_bitwidth => 9,
            inh_cnt_bitwidth => 7,
            neuron_bit_width => 8,
            shift => 4
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start_all,
            restart => restart,
            exc_spikes => exc_spikes_1_to_2,
            inh_spikes => layer_2_feedback,
            ready => layer_2_ready,
            out_spikes => layer_2_feedback
        );

    layer_3 : layer_10_neurons_128_inputs
        generic map(
            n_exc_inputs => 128,
            n_inh_inputs => 10,
            exc_cnt_bitwidth => 7,
            inh_cnt_bitwidth => 4,
            neuron_bit_width => 8,
            shift => 4
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start_all,
            restart => restart,
            exc_spikes => exc_spikes_2_to_3,
            inh_spikes => layer_3_feedback,
            ready => layer_3_ready,
            out_spikes => layer_3_feedback,
            neuron_dp_none_v => neuron_dp_none_v
        );


end architecture behavior;

