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


entity layer_10_neurons_128_inputs is
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
end entity layer_10_neurons_128_inputs;

architecture behavior of layer_10_neurons_128_inputs is


    constant v_th_0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9 : signed(neuron_bit_width-1 downto 0) := "01000000";

    component multi_input_128_exc_10_inh is
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
    end component;

    component neuron_none is
        generic (
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            v_th : in signed(neuron_bit_width-1 downto 0);
            inh_weight : in signed(neuron_bit_width-1 downto 0);
            exc_weight : in signed(neuron_bit_width-1 downto 0);
            clk : in std_logic;
            rst_n : in std_logic;
            restart : in std_logic;
            exc : in std_logic;
            inh : in std_logic;
            exc_spike : in std_logic;
            inh_spike : in std_logic;
            neuron_ready : out std_logic;
            out_spike : out std_logic;
            neuron_dp_none_v : out signed(neuron_bit_width-1 downto 0)
        );
    end component;

    component rom_128x10_exclif4 is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(6 downto 0);
            dout_0 : out std_logic_vector(7 downto 0);
            dout_1 : out std_logic_vector(7 downto 0);
            dout_2 : out std_logic_vector(7 downto 0);
            dout_3 : out std_logic_vector(7 downto 0);
            dout_4 : out std_logic_vector(7 downto 0);
            dout_5 : out std_logic_vector(7 downto 0);
            dout_6 : out std_logic_vector(7 downto 0);
            dout_7 : out std_logic_vector(7 downto 0);
            dout_8 : out std_logic_vector(7 downto 0);
            dout_9 : out std_logic_vector(7 downto 0)
        );
    end component;

    component rom_10x10_inhlif4 is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(3 downto 0);
            dout_0 : out std_logic_vector(7 downto 0);
            dout_1 : out std_logic_vector(7 downto 0);
            dout_2 : out std_logic_vector(7 downto 0);
            dout_3 : out std_logic_vector(7 downto 0);
            dout_4 : out std_logic_vector(7 downto 0);
            dout_5 : out std_logic_vector(7 downto 0);
            dout_6 : out std_logic_vector(7 downto 0);
            dout_7 : out std_logic_vector(7 downto 0);
            dout_8 : out std_logic_vector(7 downto 0);
            dout_9 : out std_logic_vector(7 downto 0)
        );
    end component;

    component addr_converter is
        generic (
            N : integer := 7
        );
        port (
            addr_in : in std_logic_vector(N-1 downto 0);
            addr_out : out std_logic_vector(N-1 downto 0)
        );
    end component;

    component barrier is
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
    end component;


    signal start_neurons : std_logic;
    signal neurons_restart : std_logic;
    signal neurons_ready : std_logic;
    signal exc : std_logic;
    signal inh : std_logic;
    signal exc_spike : std_logic;
    signal inh_spike : std_logic;
    signal exc_cnt : std_logic_vector(exc_cnt_bitwidth - 1 downto 0);
    signal inh_cnt : std_logic_vector(inh_cnt_bitwidth - 1 downto 0);
    signal exc_addr : std_logic_vector(exc_cnt_bitwidth - 1 downto 0);
    signal inh_addr : std_logic_vector(inh_cnt_bitwidth - 1 downto 0);
    signal neuron_restart : std_logic;
    signal barrier_ready : std_logic;
    signal out_spikes_inst : std_logic_vector(9 downto 0);
    signal out_sample : std_logic;
    signal neuron_ready_0 : std_logic;
    signal inh_weight_0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1 : std_logic;
    signal inh_weight_1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2 : std_logic;
    signal inh_weight_2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3 : std_logic;
    signal inh_weight_3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4 : std_logic;
    signal inh_weight_4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5 : std_logic;
    signal inh_weight_5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6 : std_logic;
    signal inh_weight_6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7 : std_logic;
    signal inh_weight_7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8 : std_logic;
    signal inh_weight_8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9 : std_logic;
    signal inh_weight_9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9 : std_logic_vector(neuron_bit_width-1 downto 0);

begin

    neurons_ready <= neuron_ready_0 and neuron_ready_1 and neuron_ready_2 and neuron_ready_3 and neuron_ready_4 and neuron_ready_5 and neuron_ready_6 and neuron_ready_7 and neuron_ready_8 and neuron_ready_9 and barrier_ready;


    multi_input_control : multi_input_128_exc_10_inh
        generic map(
            n_exc_inputs => n_exc_inputs,
            n_inh_inputs => n_inh_inputs,
            exc_cnt_bitwidth => exc_cnt_bitwidth,
            inh_cnt_bitwidth => inh_cnt_bitwidth
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            restart => restart,
            start => start,
            exc_spikes => exc_spikes,
            inh_spikes => inh_spikes,
            neurons_ready => neurons_ready,
            exc_cnt => exc_cnt,
            inh_cnt => inh_cnt,
            ready => ready,
            neuron_restart => neuron_restart,
            exc => exc,
            inh => inh,
            out_sample => out_sample,
            exc_spike => exc_spike,
            inh_spike => inh_spike
        );

    neuron_0 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0,
            inh_weight => signed(inh_weight_0),
            exc_weight => signed(exc_weight_0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0,
            out_spike => out_spikes_inst(0),
            neuron_dp_none_v => neuron_dp_none_v(7 downto 0)
        );

    neuron_1 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1,
            inh_weight => signed(inh_weight_1),
            exc_weight => signed(exc_weight_1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1,
            out_spike => out_spikes_inst(1),
            neuron_dp_none_v => neuron_dp_none_v(15 downto 8)
        );

    neuron_2 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2,
            inh_weight => signed(inh_weight_2),
            exc_weight => signed(exc_weight_2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2,
            out_spike => out_spikes_inst(2),
            neuron_dp_none_v => neuron_dp_none_v(23 downto 16)
        );

    neuron_3 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3,
            inh_weight => signed(inh_weight_3),
            exc_weight => signed(exc_weight_3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3,
            out_spike => out_spikes_inst(3),
            neuron_dp_none_v => neuron_dp_none_v(31 downto 24)
        );

    neuron_4 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4,
            inh_weight => signed(inh_weight_4),
            exc_weight => signed(exc_weight_4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4,
            out_spike => out_spikes_inst(4),
            neuron_dp_none_v => neuron_dp_none_v(39 downto 32)
        );

    neuron_5 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5,
            inh_weight => signed(inh_weight_5),
            exc_weight => signed(exc_weight_5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5,
            out_spike => out_spikes_inst(5),
            neuron_dp_none_v => neuron_dp_none_v(47 downto 40)
        );

    neuron_6 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6,
            inh_weight => signed(inh_weight_6),
            exc_weight => signed(exc_weight_6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6,
            out_spike => out_spikes_inst(6),
            neuron_dp_none_v => neuron_dp_none_v(55 downto 48)
        );

    neuron_7 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7,
            inh_weight => signed(inh_weight_7),
            exc_weight => signed(exc_weight_7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7,
            out_spike => out_spikes_inst(7),
            neuron_dp_none_v => neuron_dp_none_v(63 downto 56)
        );

    neuron_8 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8,
            inh_weight => signed(inh_weight_8),
            exc_weight => signed(exc_weight_8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8,
            out_spike => out_spikes_inst(8),
            neuron_dp_none_v => neuron_dp_none_v(71 downto 64)
        );

    neuron_9 : neuron_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9,
            inh_weight => signed(inh_weight_9),
            exc_weight => signed(exc_weight_9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9,
            out_spike => out_spikes_inst(9),
            neuron_dp_none_v => neuron_dp_none_v(79 downto 72)
        );

    exc_mem : rom_128x10_exclif4
        port map(
            clka => clk,
            addra => exc_addr,
            dout_0 => exc_weight_0,
            dout_1 => exc_weight_1,
            dout_2 => exc_weight_2,
            dout_3 => exc_weight_3,
            dout_4 => exc_weight_4,
            dout_5 => exc_weight_5,
            dout_6 => exc_weight_6,
            dout_7 => exc_weight_7,
            dout_8 => exc_weight_8,
            dout_9 => exc_weight_9
        );

    exc_addr_conv : addr_converter
        generic map(
            N => exc_cnt_bitwidth
        )
        port map(
            addr_in => exc_cnt,
            addr_out => exc_addr
        );

    inh_mem : rom_10x10_inhlif4
        port map(
            clka => clk,
            addra => inh_addr,
            dout_0 => inh_weight_0,
            dout_1 => inh_weight_1,
            dout_2 => inh_weight_2,
            dout_3 => inh_weight_3,
            dout_4 => inh_weight_4,
            dout_5 => inh_weight_5,
            dout_6 => inh_weight_6,
            dout_7 => inh_weight_7,
            dout_8 => inh_weight_8,
            dout_9 => inh_weight_9
        );

    inh_addr_conv : addr_converter
        generic map(
            N => inh_cnt_bitwidth
        )
        port map(
            addr_in => inh_cnt,
            addr_out => inh_addr
        );

    spikes_barrier : barrier
        generic map(
            N => 10
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            restart => restart,
            out_sample => out_sample,
            reg_in => out_spikes_inst,
            ready => barrier_ready,
            reg_out => out_spikes
        );


end architecture behavior;

