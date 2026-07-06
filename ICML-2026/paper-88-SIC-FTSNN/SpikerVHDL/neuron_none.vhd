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


entity neuron_none is
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
end entity neuron_none;

architecture behavior of neuron_none is


    component neuron_dp_none is
        generic (
            neuron_bit_width : integer := 8;
            shift : integer := 4
        );
        port (
            v_th : in signed(neuron_bit_width-1 downto 0);
            inh_weight : in signed(neuron_bit_width-1 downto 0);
            exc_weight : in signed(neuron_bit_width-1 downto 0);
            clk : in std_logic;
            update_sel : in std_logic_vector(1 downto 0);
            add_or_sub : in std_logic;
            v_en : in std_logic;
            v_rst_n : in std_logic;
            exceed_v_th : out std_logic;
            neuron_dp_none_v : out signed(neuron_bit_width-1 downto 0)
        );
    end component;

    component neuron_cu_none is
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            restart : in std_logic;
            exc : in std_logic;
            inh : in std_logic;
            exceed_v_th : in std_logic;
            update_sel : out std_logic_vector(1 downto 0);
            add_or_sub : out std_logic;
            v_en : out std_logic;
            v_rst_n : out std_logic;
            neuron_ready : out std_logic;
            out_spike : out std_logic
        );
    end component;

    component and_mask is
        generic (
            N : integer := 8
        );
        port (
            input_bits : in signed(N-1 downto 0);
            mask_bit : in std_logic;
            output_bits : out signed(N-1 downto 0)
        );
    end component;


    signal update_sel : std_logic_vector(1 downto 0);
    signal add_or_sub : std_logic;
    signal v_en : std_logic;
    signal v_rst_n : std_logic;
    signal exceed_v_th : std_logic;
    signal masked_inh_weight : signed(neuron_bit_width-1 downto 0);
    signal masked_exc_weight : signed(neuron_bit_width-1 downto 0);

begin

    datapath : neuron_dp_none
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th,
            inh_weight => masked_inh_weight,
            exc_weight => masked_exc_weight,
            clk => clk,
            update_sel => update_sel,
            add_or_sub => add_or_sub,
            v_en => v_en,
            v_rst_n => v_rst_n,
            exceed_v_th => exceed_v_th,
            neuron_dp_none_v => neuron_dp_none_v
        );

    control_unit : neuron_cu_none
        port map(
            clk => clk,
            rst_n => rst_n,
            restart => restart,
            exc => exc,
            inh => inh,
            exceed_v_th => exceed_v_th,
            update_sel => update_sel,
            add_or_sub => add_or_sub,
            v_en => v_en,
            v_rst_n => v_rst_n,
            neuron_ready => neuron_ready,
            out_spike => out_spike
        );

    exc_mask : and_mask
        generic map(
            N => neuron_bit_width
        )
        port map(
            input_bits => exc_weight,
            mask_bit => exc_spike,
            output_bits => masked_exc_weight
        );

    inh_mask : and_mask
        generic map(
            N => neuron_bit_width
        )
        port map(
            input_bits => inh_weight,
            mask_bit => inh_spike,
            output_bits => masked_inh_weight
        );


end architecture behavior;

