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


entity neuron_dp_subtractive is
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
        exceed_v_th : out std_logic
    );
end entity neuron_dp_subtractive;

architecture behavior of neuron_dp_subtractive is


    component shifter is
        generic (
            N : integer := 8;
            shift : integer := 4
        );
        port (
            shifter_in : in signed(N-1 downto 0);
            shifted_out : out signed(N-1 downto 0)
        );
    end component;

    component add_sub is
        generic (
            N : integer := 8
        );
        port (
            in0 : in signed(N-1 downto 0);
            in1 : in signed(N-1 downto 0);
            add_or_sub : in std_logic;
            add_sub_out : out signed(N-1 downto 0)
        );
    end component;

    component mux_4to1_signed is
        generic (
            bitwidth : integer := 8
        );
        port (
            mux_sel : in std_logic_vector(1 downto 0);
            in0 : in signed(bitwidth-1 downto 0);
            in1 : in signed(bitwidth-1 downto 0);
            in2 : in signed(bitwidth-1 downto 0);
            in3 : in signed(bitwidth-1 downto 0);
            mux_out : out signed(bitwidth-1 downto 0)
        );
    end component;

    component reg_signed_sync_rst is
        generic (
            N : integer := 8
        );
        port (
            clk : in std_logic;
            en : in std_logic;
            rst_n : in std_logic;
            reg_in : in signed(N-1 downto 0);
            reg_out : out signed(N-1 downto 0)
        );
    end component;

    component cmp_gt is
        generic (
            N : integer := 8
        );
        port (
            in0 : in signed(N-1 downto 0);
            in1 : in signed(N-1 downto 0);
            cmp_out : out std_logic
        );
    end component;


    signal update : signed(neuron_bit_width-1 downto 0);
    signal v_value : signed(neuron_bit_width-1 downto 0);
    signal v : signed(neuron_bit_width-1 downto 0);
    signal v_shifted : signed(neuron_bit_width-1 downto 0);

begin

    v_shifter : shifter
        generic map(
            N => neuron_bit_width,
            shift => shift
        )
        port map(
            shifter_in => v,
            shifted_out => v_shifted
        );

    update_mux : mux_4to1_signed
        generic map(
            bitwidth => neuron_bit_width
        )
        port map(
            mux_sel => update_sel,
            in0 => v_th,
            in1 => v_shifted,
            in2 => exc_weight,
            in3 => inh_weight,
            mux_out => update
        );

    update_add_sub : add_sub
        generic map(
            N => neuron_bit_width
        )
        port map(
            in0 => v,
            in1 => update,
            add_or_sub => add_or_sub,
            add_sub_out => v_value
        );

    v_reg : reg_signed_sync_rst
        generic map(
            N => neuron_bit_width
        )
        port map(
            clk => clk,
            en => v_en,
            rst_n => v_rst_n,
            reg_in => v_value,
            reg_out => v
        );

    fire_cmp : cmp_gt
        generic map(
            N => neuron_bit_width
        )
        port map(
            in0 => v,
            in1 => v_th,
            cmp_out => exceed_v_th
        );


end architecture behavior;

