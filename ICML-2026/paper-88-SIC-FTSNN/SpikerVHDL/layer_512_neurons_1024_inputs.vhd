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


entity layer_512_neurons_1024_inputs is
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
end entity layer_512_neurons_1024_inputs;

architecture behavior of layer_512_neurons_1024_inputs is


    constant v_th_00 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_01 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_02 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_03 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_04 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_05 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_06 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_07 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_08 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_09 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_0f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_20 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_21 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_22 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_23 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_24 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_25 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_26 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_27 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_28 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_29 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_2f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_30 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_31 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_32 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_33 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_34 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_35 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_36 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_37 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_38 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_39 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_3f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_40 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_41 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_42 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_43 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_44 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_45 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_46 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_47 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_48 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_49 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_4f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_50 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_51 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_52 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_53 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_54 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_55 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_56 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_57 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_58 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_59 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_5f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_60 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_61 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_62 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_63 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_64 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_65 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_66 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_67 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_68 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_69 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_6f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_70 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_71 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_72 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_73 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_74 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_75 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_76 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_77 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_78 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_79 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_7f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_80 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_81 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_82 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_83 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_84 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_85 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_86 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_87 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_88 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_89 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_8f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_90 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_91 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_92 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_93 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_94 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_95 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_96 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_97 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_98 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_99 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_9f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_a9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_aa : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ab : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ac : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ad : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ae : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_af : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_b9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ba : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_bb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_bc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_bd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_be : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_bf : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_c9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ca : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_cb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_cc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_cd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ce : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_cf : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_d9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_da : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_db : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_dc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_dd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_de : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_df : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_e9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ea : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_eb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ec : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ed : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ee : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ef : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_f9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_fa : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_fb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_fc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_fd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_fe : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_ff : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_100 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_101 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_102 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_103 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_104 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_105 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_106 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_107 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_108 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_109 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_10f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_110 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_111 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_112 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_113 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_114 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_115 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_116 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_117 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_118 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_119 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_11f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_120 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_121 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_122 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_123 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_124 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_125 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_126 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_127 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_128 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_129 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_12f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_130 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_131 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_132 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_133 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_134 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_135 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_136 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_137 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_138 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_139 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_13f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_140 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_141 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_142 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_143 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_144 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_145 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_146 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_147 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_148 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_149 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_14f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_150 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_151 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_152 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_153 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_154 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_155 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_156 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_157 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_158 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_159 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_15f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_160 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_161 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_162 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_163 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_164 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_165 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_166 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_167 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_168 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_169 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_16f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_170 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_171 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_172 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_173 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_174 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_175 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_176 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_177 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_178 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_179 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_17f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_180 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_181 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_182 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_183 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_184 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_185 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_186 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_187 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_188 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_189 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_18f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_190 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_191 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_192 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_193 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_194 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_195 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_196 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_197 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_198 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_199 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19a : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19b : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19c : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19d : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19e : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_19f : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1a9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1aa : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ab : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ac : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ad : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ae : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1af : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1b9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ba : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1bb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1bc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1bd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1be : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1bf : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1c9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ca : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1cb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1cc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1cd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ce : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1cf : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1d9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1da : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1db : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1dc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1dd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1de : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1df : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1e9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ea : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1eb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ec : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ed : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ee : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ef : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f0 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f1 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f2 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f3 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f4 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f5 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f6 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f7 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f8 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1f9 : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1fa : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1fb : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1fc : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1fd : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1fe : signed(neuron_bit_width-1 downto 0) := "01000000";
    constant v_th_1ff : signed(neuron_bit_width-1 downto 0) := "01000000";

    component multi_input_1024_exc_512_inh is
        generic (
            n_exc_inputs : integer := 1024;
            n_inh_inputs : integer := 512;
            exc_cnt_bitwidth : integer := 10;
            inh_cnt_bitwidth : integer := 9
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

    component neuron_subtractive is
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
            out_spike : out std_logic
        );
    end component;

    component rom_1024x512_exclif2 is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(9 downto 0);
            dout_00 : out std_logic_vector(7 downto 0);
            dout_01 : out std_logic_vector(7 downto 0);
            dout_02 : out std_logic_vector(7 downto 0);
            dout_03 : out std_logic_vector(7 downto 0);
            dout_04 : out std_logic_vector(7 downto 0);
            dout_05 : out std_logic_vector(7 downto 0);
            dout_06 : out std_logic_vector(7 downto 0);
            dout_07 : out std_logic_vector(7 downto 0);
            dout_08 : out std_logic_vector(7 downto 0);
            dout_09 : out std_logic_vector(7 downto 0);
            dout_0a : out std_logic_vector(7 downto 0);
            dout_0b : out std_logic_vector(7 downto 0);
            dout_0c : out std_logic_vector(7 downto 0);
            dout_0d : out std_logic_vector(7 downto 0);
            dout_0e : out std_logic_vector(7 downto 0);
            dout_0f : out std_logic_vector(7 downto 0);
            dout_10 : out std_logic_vector(7 downto 0);
            dout_11 : out std_logic_vector(7 downto 0);
            dout_12 : out std_logic_vector(7 downto 0);
            dout_13 : out std_logic_vector(7 downto 0);
            dout_14 : out std_logic_vector(7 downto 0);
            dout_15 : out std_logic_vector(7 downto 0);
            dout_16 : out std_logic_vector(7 downto 0);
            dout_17 : out std_logic_vector(7 downto 0);
            dout_18 : out std_logic_vector(7 downto 0);
            dout_19 : out std_logic_vector(7 downto 0);
            dout_1a : out std_logic_vector(7 downto 0);
            dout_1b : out std_logic_vector(7 downto 0);
            dout_1c : out std_logic_vector(7 downto 0);
            dout_1d : out std_logic_vector(7 downto 0);
            dout_1e : out std_logic_vector(7 downto 0);
            dout_1f : out std_logic_vector(7 downto 0);
            dout_20 : out std_logic_vector(7 downto 0);
            dout_21 : out std_logic_vector(7 downto 0);
            dout_22 : out std_logic_vector(7 downto 0);
            dout_23 : out std_logic_vector(7 downto 0);
            dout_24 : out std_logic_vector(7 downto 0);
            dout_25 : out std_logic_vector(7 downto 0);
            dout_26 : out std_logic_vector(7 downto 0);
            dout_27 : out std_logic_vector(7 downto 0);
            dout_28 : out std_logic_vector(7 downto 0);
            dout_29 : out std_logic_vector(7 downto 0);
            dout_2a : out std_logic_vector(7 downto 0);
            dout_2b : out std_logic_vector(7 downto 0);
            dout_2c : out std_logic_vector(7 downto 0);
            dout_2d : out std_logic_vector(7 downto 0);
            dout_2e : out std_logic_vector(7 downto 0);
            dout_2f : out std_logic_vector(7 downto 0);
            dout_30 : out std_logic_vector(7 downto 0);
            dout_31 : out std_logic_vector(7 downto 0);
            dout_32 : out std_logic_vector(7 downto 0);
            dout_33 : out std_logic_vector(7 downto 0);
            dout_34 : out std_logic_vector(7 downto 0);
            dout_35 : out std_logic_vector(7 downto 0);
            dout_36 : out std_logic_vector(7 downto 0);
            dout_37 : out std_logic_vector(7 downto 0);
            dout_38 : out std_logic_vector(7 downto 0);
            dout_39 : out std_logic_vector(7 downto 0);
            dout_3a : out std_logic_vector(7 downto 0);
            dout_3b : out std_logic_vector(7 downto 0);
            dout_3c : out std_logic_vector(7 downto 0);
            dout_3d : out std_logic_vector(7 downto 0);
            dout_3e : out std_logic_vector(7 downto 0);
            dout_3f : out std_logic_vector(7 downto 0);
            dout_40 : out std_logic_vector(7 downto 0);
            dout_41 : out std_logic_vector(7 downto 0);
            dout_42 : out std_logic_vector(7 downto 0);
            dout_43 : out std_logic_vector(7 downto 0);
            dout_44 : out std_logic_vector(7 downto 0);
            dout_45 : out std_logic_vector(7 downto 0);
            dout_46 : out std_logic_vector(7 downto 0);
            dout_47 : out std_logic_vector(7 downto 0);
            dout_48 : out std_logic_vector(7 downto 0);
            dout_49 : out std_logic_vector(7 downto 0);
            dout_4a : out std_logic_vector(7 downto 0);
            dout_4b : out std_logic_vector(7 downto 0);
            dout_4c : out std_logic_vector(7 downto 0);
            dout_4d : out std_logic_vector(7 downto 0);
            dout_4e : out std_logic_vector(7 downto 0);
            dout_4f : out std_logic_vector(7 downto 0);
            dout_50 : out std_logic_vector(7 downto 0);
            dout_51 : out std_logic_vector(7 downto 0);
            dout_52 : out std_logic_vector(7 downto 0);
            dout_53 : out std_logic_vector(7 downto 0);
            dout_54 : out std_logic_vector(7 downto 0);
            dout_55 : out std_logic_vector(7 downto 0);
            dout_56 : out std_logic_vector(7 downto 0);
            dout_57 : out std_logic_vector(7 downto 0);
            dout_58 : out std_logic_vector(7 downto 0);
            dout_59 : out std_logic_vector(7 downto 0);
            dout_5a : out std_logic_vector(7 downto 0);
            dout_5b : out std_logic_vector(7 downto 0);
            dout_5c : out std_logic_vector(7 downto 0);
            dout_5d : out std_logic_vector(7 downto 0);
            dout_5e : out std_logic_vector(7 downto 0);
            dout_5f : out std_logic_vector(7 downto 0);
            dout_60 : out std_logic_vector(7 downto 0);
            dout_61 : out std_logic_vector(7 downto 0);
            dout_62 : out std_logic_vector(7 downto 0);
            dout_63 : out std_logic_vector(7 downto 0);
            dout_64 : out std_logic_vector(7 downto 0);
            dout_65 : out std_logic_vector(7 downto 0);
            dout_66 : out std_logic_vector(7 downto 0);
            dout_67 : out std_logic_vector(7 downto 0);
            dout_68 : out std_logic_vector(7 downto 0);
            dout_69 : out std_logic_vector(7 downto 0);
            dout_6a : out std_logic_vector(7 downto 0);
            dout_6b : out std_logic_vector(7 downto 0);
            dout_6c : out std_logic_vector(7 downto 0);
            dout_6d : out std_logic_vector(7 downto 0);
            dout_6e : out std_logic_vector(7 downto 0);
            dout_6f : out std_logic_vector(7 downto 0);
            dout_70 : out std_logic_vector(7 downto 0);
            dout_71 : out std_logic_vector(7 downto 0);
            dout_72 : out std_logic_vector(7 downto 0);
            dout_73 : out std_logic_vector(7 downto 0);
            dout_74 : out std_logic_vector(7 downto 0);
            dout_75 : out std_logic_vector(7 downto 0);
            dout_76 : out std_logic_vector(7 downto 0);
            dout_77 : out std_logic_vector(7 downto 0);
            dout_78 : out std_logic_vector(7 downto 0);
            dout_79 : out std_logic_vector(7 downto 0);
            dout_7a : out std_logic_vector(7 downto 0);
            dout_7b : out std_logic_vector(7 downto 0);
            dout_7c : out std_logic_vector(7 downto 0);
            dout_7d : out std_logic_vector(7 downto 0);
            dout_7e : out std_logic_vector(7 downto 0);
            dout_7f : out std_logic_vector(7 downto 0);
            dout_80 : out std_logic_vector(7 downto 0);
            dout_81 : out std_logic_vector(7 downto 0);
            dout_82 : out std_logic_vector(7 downto 0);
            dout_83 : out std_logic_vector(7 downto 0);
            dout_84 : out std_logic_vector(7 downto 0);
            dout_85 : out std_logic_vector(7 downto 0);
            dout_86 : out std_logic_vector(7 downto 0);
            dout_87 : out std_logic_vector(7 downto 0);
            dout_88 : out std_logic_vector(7 downto 0);
            dout_89 : out std_logic_vector(7 downto 0);
            dout_8a : out std_logic_vector(7 downto 0);
            dout_8b : out std_logic_vector(7 downto 0);
            dout_8c : out std_logic_vector(7 downto 0);
            dout_8d : out std_logic_vector(7 downto 0);
            dout_8e : out std_logic_vector(7 downto 0);
            dout_8f : out std_logic_vector(7 downto 0);
            dout_90 : out std_logic_vector(7 downto 0);
            dout_91 : out std_logic_vector(7 downto 0);
            dout_92 : out std_logic_vector(7 downto 0);
            dout_93 : out std_logic_vector(7 downto 0);
            dout_94 : out std_logic_vector(7 downto 0);
            dout_95 : out std_logic_vector(7 downto 0);
            dout_96 : out std_logic_vector(7 downto 0);
            dout_97 : out std_logic_vector(7 downto 0);
            dout_98 : out std_logic_vector(7 downto 0);
            dout_99 : out std_logic_vector(7 downto 0);
            dout_9a : out std_logic_vector(7 downto 0);
            dout_9b : out std_logic_vector(7 downto 0);
            dout_9c : out std_logic_vector(7 downto 0);
            dout_9d : out std_logic_vector(7 downto 0);
            dout_9e : out std_logic_vector(7 downto 0);
            dout_9f : out std_logic_vector(7 downto 0);
            dout_a0 : out std_logic_vector(7 downto 0);
            dout_a1 : out std_logic_vector(7 downto 0);
            dout_a2 : out std_logic_vector(7 downto 0);
            dout_a3 : out std_logic_vector(7 downto 0);
            dout_a4 : out std_logic_vector(7 downto 0);
            dout_a5 : out std_logic_vector(7 downto 0);
            dout_a6 : out std_logic_vector(7 downto 0);
            dout_a7 : out std_logic_vector(7 downto 0);
            dout_a8 : out std_logic_vector(7 downto 0);
            dout_a9 : out std_logic_vector(7 downto 0);
            dout_aa : out std_logic_vector(7 downto 0);
            dout_ab : out std_logic_vector(7 downto 0);
            dout_ac : out std_logic_vector(7 downto 0);
            dout_ad : out std_logic_vector(7 downto 0);
            dout_ae : out std_logic_vector(7 downto 0);
            dout_af : out std_logic_vector(7 downto 0);
            dout_b0 : out std_logic_vector(7 downto 0);
            dout_b1 : out std_logic_vector(7 downto 0);
            dout_b2 : out std_logic_vector(7 downto 0);
            dout_b3 : out std_logic_vector(7 downto 0);
            dout_b4 : out std_logic_vector(7 downto 0);
            dout_b5 : out std_logic_vector(7 downto 0);
            dout_b6 : out std_logic_vector(7 downto 0);
            dout_b7 : out std_logic_vector(7 downto 0);
            dout_b8 : out std_logic_vector(7 downto 0);
            dout_b9 : out std_logic_vector(7 downto 0);
            dout_ba : out std_logic_vector(7 downto 0);
            dout_bb : out std_logic_vector(7 downto 0);
            dout_bc : out std_logic_vector(7 downto 0);
            dout_bd : out std_logic_vector(7 downto 0);
            dout_be : out std_logic_vector(7 downto 0);
            dout_bf : out std_logic_vector(7 downto 0);
            dout_c0 : out std_logic_vector(7 downto 0);
            dout_c1 : out std_logic_vector(7 downto 0);
            dout_c2 : out std_logic_vector(7 downto 0);
            dout_c3 : out std_logic_vector(7 downto 0);
            dout_c4 : out std_logic_vector(7 downto 0);
            dout_c5 : out std_logic_vector(7 downto 0);
            dout_c6 : out std_logic_vector(7 downto 0);
            dout_c7 : out std_logic_vector(7 downto 0);
            dout_c8 : out std_logic_vector(7 downto 0);
            dout_c9 : out std_logic_vector(7 downto 0);
            dout_ca : out std_logic_vector(7 downto 0);
            dout_cb : out std_logic_vector(7 downto 0);
            dout_cc : out std_logic_vector(7 downto 0);
            dout_cd : out std_logic_vector(7 downto 0);
            dout_ce : out std_logic_vector(7 downto 0);
            dout_cf : out std_logic_vector(7 downto 0);
            dout_d0 : out std_logic_vector(7 downto 0);
            dout_d1 : out std_logic_vector(7 downto 0);
            dout_d2 : out std_logic_vector(7 downto 0);
            dout_d3 : out std_logic_vector(7 downto 0);
            dout_d4 : out std_logic_vector(7 downto 0);
            dout_d5 : out std_logic_vector(7 downto 0);
            dout_d6 : out std_logic_vector(7 downto 0);
            dout_d7 : out std_logic_vector(7 downto 0);
            dout_d8 : out std_logic_vector(7 downto 0);
            dout_d9 : out std_logic_vector(7 downto 0);
            dout_da : out std_logic_vector(7 downto 0);
            dout_db : out std_logic_vector(7 downto 0);
            dout_dc : out std_logic_vector(7 downto 0);
            dout_dd : out std_logic_vector(7 downto 0);
            dout_de : out std_logic_vector(7 downto 0);
            dout_df : out std_logic_vector(7 downto 0);
            dout_e0 : out std_logic_vector(7 downto 0);
            dout_e1 : out std_logic_vector(7 downto 0);
            dout_e2 : out std_logic_vector(7 downto 0);
            dout_e3 : out std_logic_vector(7 downto 0);
            dout_e4 : out std_logic_vector(7 downto 0);
            dout_e5 : out std_logic_vector(7 downto 0);
            dout_e6 : out std_logic_vector(7 downto 0);
            dout_e7 : out std_logic_vector(7 downto 0);
            dout_e8 : out std_logic_vector(7 downto 0);
            dout_e9 : out std_logic_vector(7 downto 0);
            dout_ea : out std_logic_vector(7 downto 0);
            dout_eb : out std_logic_vector(7 downto 0);
            dout_ec : out std_logic_vector(7 downto 0);
            dout_ed : out std_logic_vector(7 downto 0);
            dout_ee : out std_logic_vector(7 downto 0);
            dout_ef : out std_logic_vector(7 downto 0);
            dout_f0 : out std_logic_vector(7 downto 0);
            dout_f1 : out std_logic_vector(7 downto 0);
            dout_f2 : out std_logic_vector(7 downto 0);
            dout_f3 : out std_logic_vector(7 downto 0);
            dout_f4 : out std_logic_vector(7 downto 0);
            dout_f5 : out std_logic_vector(7 downto 0);
            dout_f6 : out std_logic_vector(7 downto 0);
            dout_f7 : out std_logic_vector(7 downto 0);
            dout_f8 : out std_logic_vector(7 downto 0);
            dout_f9 : out std_logic_vector(7 downto 0);
            dout_fa : out std_logic_vector(7 downto 0);
            dout_fb : out std_logic_vector(7 downto 0);
            dout_fc : out std_logic_vector(7 downto 0);
            dout_fd : out std_logic_vector(7 downto 0);
            dout_fe : out std_logic_vector(7 downto 0);
            dout_ff : out std_logic_vector(7 downto 0);
            dout_100 : out std_logic_vector(7 downto 0);
            dout_101 : out std_logic_vector(7 downto 0);
            dout_102 : out std_logic_vector(7 downto 0);
            dout_103 : out std_logic_vector(7 downto 0);
            dout_104 : out std_logic_vector(7 downto 0);
            dout_105 : out std_logic_vector(7 downto 0);
            dout_106 : out std_logic_vector(7 downto 0);
            dout_107 : out std_logic_vector(7 downto 0);
            dout_108 : out std_logic_vector(7 downto 0);
            dout_109 : out std_logic_vector(7 downto 0);
            dout_10a : out std_logic_vector(7 downto 0);
            dout_10b : out std_logic_vector(7 downto 0);
            dout_10c : out std_logic_vector(7 downto 0);
            dout_10d : out std_logic_vector(7 downto 0);
            dout_10e : out std_logic_vector(7 downto 0);
            dout_10f : out std_logic_vector(7 downto 0);
            dout_110 : out std_logic_vector(7 downto 0);
            dout_111 : out std_logic_vector(7 downto 0);
            dout_112 : out std_logic_vector(7 downto 0);
            dout_113 : out std_logic_vector(7 downto 0);
            dout_114 : out std_logic_vector(7 downto 0);
            dout_115 : out std_logic_vector(7 downto 0);
            dout_116 : out std_logic_vector(7 downto 0);
            dout_117 : out std_logic_vector(7 downto 0);
            dout_118 : out std_logic_vector(7 downto 0);
            dout_119 : out std_logic_vector(7 downto 0);
            dout_11a : out std_logic_vector(7 downto 0);
            dout_11b : out std_logic_vector(7 downto 0);
            dout_11c : out std_logic_vector(7 downto 0);
            dout_11d : out std_logic_vector(7 downto 0);
            dout_11e : out std_logic_vector(7 downto 0);
            dout_11f : out std_logic_vector(7 downto 0);
            dout_120 : out std_logic_vector(7 downto 0);
            dout_121 : out std_logic_vector(7 downto 0);
            dout_122 : out std_logic_vector(7 downto 0);
            dout_123 : out std_logic_vector(7 downto 0);
            dout_124 : out std_logic_vector(7 downto 0);
            dout_125 : out std_logic_vector(7 downto 0);
            dout_126 : out std_logic_vector(7 downto 0);
            dout_127 : out std_logic_vector(7 downto 0);
            dout_128 : out std_logic_vector(7 downto 0);
            dout_129 : out std_logic_vector(7 downto 0);
            dout_12a : out std_logic_vector(7 downto 0);
            dout_12b : out std_logic_vector(7 downto 0);
            dout_12c : out std_logic_vector(7 downto 0);
            dout_12d : out std_logic_vector(7 downto 0);
            dout_12e : out std_logic_vector(7 downto 0);
            dout_12f : out std_logic_vector(7 downto 0);
            dout_130 : out std_logic_vector(7 downto 0);
            dout_131 : out std_logic_vector(7 downto 0);
            dout_132 : out std_logic_vector(7 downto 0);
            dout_133 : out std_logic_vector(7 downto 0);
            dout_134 : out std_logic_vector(7 downto 0);
            dout_135 : out std_logic_vector(7 downto 0);
            dout_136 : out std_logic_vector(7 downto 0);
            dout_137 : out std_logic_vector(7 downto 0);
            dout_138 : out std_logic_vector(7 downto 0);
            dout_139 : out std_logic_vector(7 downto 0);
            dout_13a : out std_logic_vector(7 downto 0);
            dout_13b : out std_logic_vector(7 downto 0);
            dout_13c : out std_logic_vector(7 downto 0);
            dout_13d : out std_logic_vector(7 downto 0);
            dout_13e : out std_logic_vector(7 downto 0);
            dout_13f : out std_logic_vector(7 downto 0);
            dout_140 : out std_logic_vector(7 downto 0);
            dout_141 : out std_logic_vector(7 downto 0);
            dout_142 : out std_logic_vector(7 downto 0);
            dout_143 : out std_logic_vector(7 downto 0);
            dout_144 : out std_logic_vector(7 downto 0);
            dout_145 : out std_logic_vector(7 downto 0);
            dout_146 : out std_logic_vector(7 downto 0);
            dout_147 : out std_logic_vector(7 downto 0);
            dout_148 : out std_logic_vector(7 downto 0);
            dout_149 : out std_logic_vector(7 downto 0);
            dout_14a : out std_logic_vector(7 downto 0);
            dout_14b : out std_logic_vector(7 downto 0);
            dout_14c : out std_logic_vector(7 downto 0);
            dout_14d : out std_logic_vector(7 downto 0);
            dout_14e : out std_logic_vector(7 downto 0);
            dout_14f : out std_logic_vector(7 downto 0);
            dout_150 : out std_logic_vector(7 downto 0);
            dout_151 : out std_logic_vector(7 downto 0);
            dout_152 : out std_logic_vector(7 downto 0);
            dout_153 : out std_logic_vector(7 downto 0);
            dout_154 : out std_logic_vector(7 downto 0);
            dout_155 : out std_logic_vector(7 downto 0);
            dout_156 : out std_logic_vector(7 downto 0);
            dout_157 : out std_logic_vector(7 downto 0);
            dout_158 : out std_logic_vector(7 downto 0);
            dout_159 : out std_logic_vector(7 downto 0);
            dout_15a : out std_logic_vector(7 downto 0);
            dout_15b : out std_logic_vector(7 downto 0);
            dout_15c : out std_logic_vector(7 downto 0);
            dout_15d : out std_logic_vector(7 downto 0);
            dout_15e : out std_logic_vector(7 downto 0);
            dout_15f : out std_logic_vector(7 downto 0);
            dout_160 : out std_logic_vector(7 downto 0);
            dout_161 : out std_logic_vector(7 downto 0);
            dout_162 : out std_logic_vector(7 downto 0);
            dout_163 : out std_logic_vector(7 downto 0);
            dout_164 : out std_logic_vector(7 downto 0);
            dout_165 : out std_logic_vector(7 downto 0);
            dout_166 : out std_logic_vector(7 downto 0);
            dout_167 : out std_logic_vector(7 downto 0);
            dout_168 : out std_logic_vector(7 downto 0);
            dout_169 : out std_logic_vector(7 downto 0);
            dout_16a : out std_logic_vector(7 downto 0);
            dout_16b : out std_logic_vector(7 downto 0);
            dout_16c : out std_logic_vector(7 downto 0);
            dout_16d : out std_logic_vector(7 downto 0);
            dout_16e : out std_logic_vector(7 downto 0);
            dout_16f : out std_logic_vector(7 downto 0);
            dout_170 : out std_logic_vector(7 downto 0);
            dout_171 : out std_logic_vector(7 downto 0);
            dout_172 : out std_logic_vector(7 downto 0);
            dout_173 : out std_logic_vector(7 downto 0);
            dout_174 : out std_logic_vector(7 downto 0);
            dout_175 : out std_logic_vector(7 downto 0);
            dout_176 : out std_logic_vector(7 downto 0);
            dout_177 : out std_logic_vector(7 downto 0);
            dout_178 : out std_logic_vector(7 downto 0);
            dout_179 : out std_logic_vector(7 downto 0);
            dout_17a : out std_logic_vector(7 downto 0);
            dout_17b : out std_logic_vector(7 downto 0);
            dout_17c : out std_logic_vector(7 downto 0);
            dout_17d : out std_logic_vector(7 downto 0);
            dout_17e : out std_logic_vector(7 downto 0);
            dout_17f : out std_logic_vector(7 downto 0);
            dout_180 : out std_logic_vector(7 downto 0);
            dout_181 : out std_logic_vector(7 downto 0);
            dout_182 : out std_logic_vector(7 downto 0);
            dout_183 : out std_logic_vector(7 downto 0);
            dout_184 : out std_logic_vector(7 downto 0);
            dout_185 : out std_logic_vector(7 downto 0);
            dout_186 : out std_logic_vector(7 downto 0);
            dout_187 : out std_logic_vector(7 downto 0);
            dout_188 : out std_logic_vector(7 downto 0);
            dout_189 : out std_logic_vector(7 downto 0);
            dout_18a : out std_logic_vector(7 downto 0);
            dout_18b : out std_logic_vector(7 downto 0);
            dout_18c : out std_logic_vector(7 downto 0);
            dout_18d : out std_logic_vector(7 downto 0);
            dout_18e : out std_logic_vector(7 downto 0);
            dout_18f : out std_logic_vector(7 downto 0);
            dout_190 : out std_logic_vector(7 downto 0);
            dout_191 : out std_logic_vector(7 downto 0);
            dout_192 : out std_logic_vector(7 downto 0);
            dout_193 : out std_logic_vector(7 downto 0);
            dout_194 : out std_logic_vector(7 downto 0);
            dout_195 : out std_logic_vector(7 downto 0);
            dout_196 : out std_logic_vector(7 downto 0);
            dout_197 : out std_logic_vector(7 downto 0);
            dout_198 : out std_logic_vector(7 downto 0);
            dout_199 : out std_logic_vector(7 downto 0);
            dout_19a : out std_logic_vector(7 downto 0);
            dout_19b : out std_logic_vector(7 downto 0);
            dout_19c : out std_logic_vector(7 downto 0);
            dout_19d : out std_logic_vector(7 downto 0);
            dout_19e : out std_logic_vector(7 downto 0);
            dout_19f : out std_logic_vector(7 downto 0);
            dout_1a0 : out std_logic_vector(7 downto 0);
            dout_1a1 : out std_logic_vector(7 downto 0);
            dout_1a2 : out std_logic_vector(7 downto 0);
            dout_1a3 : out std_logic_vector(7 downto 0);
            dout_1a4 : out std_logic_vector(7 downto 0);
            dout_1a5 : out std_logic_vector(7 downto 0);
            dout_1a6 : out std_logic_vector(7 downto 0);
            dout_1a7 : out std_logic_vector(7 downto 0);
            dout_1a8 : out std_logic_vector(7 downto 0);
            dout_1a9 : out std_logic_vector(7 downto 0);
            dout_1aa : out std_logic_vector(7 downto 0);
            dout_1ab : out std_logic_vector(7 downto 0);
            dout_1ac : out std_logic_vector(7 downto 0);
            dout_1ad : out std_logic_vector(7 downto 0);
            dout_1ae : out std_logic_vector(7 downto 0);
            dout_1af : out std_logic_vector(7 downto 0);
            dout_1b0 : out std_logic_vector(7 downto 0);
            dout_1b1 : out std_logic_vector(7 downto 0);
            dout_1b2 : out std_logic_vector(7 downto 0);
            dout_1b3 : out std_logic_vector(7 downto 0);
            dout_1b4 : out std_logic_vector(7 downto 0);
            dout_1b5 : out std_logic_vector(7 downto 0);
            dout_1b6 : out std_logic_vector(7 downto 0);
            dout_1b7 : out std_logic_vector(7 downto 0);
            dout_1b8 : out std_logic_vector(7 downto 0);
            dout_1b9 : out std_logic_vector(7 downto 0);
            dout_1ba : out std_logic_vector(7 downto 0);
            dout_1bb : out std_logic_vector(7 downto 0);
            dout_1bc : out std_logic_vector(7 downto 0);
            dout_1bd : out std_logic_vector(7 downto 0);
            dout_1be : out std_logic_vector(7 downto 0);
            dout_1bf : out std_logic_vector(7 downto 0);
            dout_1c0 : out std_logic_vector(7 downto 0);
            dout_1c1 : out std_logic_vector(7 downto 0);
            dout_1c2 : out std_logic_vector(7 downto 0);
            dout_1c3 : out std_logic_vector(7 downto 0);
            dout_1c4 : out std_logic_vector(7 downto 0);
            dout_1c5 : out std_logic_vector(7 downto 0);
            dout_1c6 : out std_logic_vector(7 downto 0);
            dout_1c7 : out std_logic_vector(7 downto 0);
            dout_1c8 : out std_logic_vector(7 downto 0);
            dout_1c9 : out std_logic_vector(7 downto 0);
            dout_1ca : out std_logic_vector(7 downto 0);
            dout_1cb : out std_logic_vector(7 downto 0);
            dout_1cc : out std_logic_vector(7 downto 0);
            dout_1cd : out std_logic_vector(7 downto 0);
            dout_1ce : out std_logic_vector(7 downto 0);
            dout_1cf : out std_logic_vector(7 downto 0);
            dout_1d0 : out std_logic_vector(7 downto 0);
            dout_1d1 : out std_logic_vector(7 downto 0);
            dout_1d2 : out std_logic_vector(7 downto 0);
            dout_1d3 : out std_logic_vector(7 downto 0);
            dout_1d4 : out std_logic_vector(7 downto 0);
            dout_1d5 : out std_logic_vector(7 downto 0);
            dout_1d6 : out std_logic_vector(7 downto 0);
            dout_1d7 : out std_logic_vector(7 downto 0);
            dout_1d8 : out std_logic_vector(7 downto 0);
            dout_1d9 : out std_logic_vector(7 downto 0);
            dout_1da : out std_logic_vector(7 downto 0);
            dout_1db : out std_logic_vector(7 downto 0);
            dout_1dc : out std_logic_vector(7 downto 0);
            dout_1dd : out std_logic_vector(7 downto 0);
            dout_1de : out std_logic_vector(7 downto 0);
            dout_1df : out std_logic_vector(7 downto 0);
            dout_1e0 : out std_logic_vector(7 downto 0);
            dout_1e1 : out std_logic_vector(7 downto 0);
            dout_1e2 : out std_logic_vector(7 downto 0);
            dout_1e3 : out std_logic_vector(7 downto 0);
            dout_1e4 : out std_logic_vector(7 downto 0);
            dout_1e5 : out std_logic_vector(7 downto 0);
            dout_1e6 : out std_logic_vector(7 downto 0);
            dout_1e7 : out std_logic_vector(7 downto 0);
            dout_1e8 : out std_logic_vector(7 downto 0);
            dout_1e9 : out std_logic_vector(7 downto 0);
            dout_1ea : out std_logic_vector(7 downto 0);
            dout_1eb : out std_logic_vector(7 downto 0);
            dout_1ec : out std_logic_vector(7 downto 0);
            dout_1ed : out std_logic_vector(7 downto 0);
            dout_1ee : out std_logic_vector(7 downto 0);
            dout_1ef : out std_logic_vector(7 downto 0);
            dout_1f0 : out std_logic_vector(7 downto 0);
            dout_1f1 : out std_logic_vector(7 downto 0);
            dout_1f2 : out std_logic_vector(7 downto 0);
            dout_1f3 : out std_logic_vector(7 downto 0);
            dout_1f4 : out std_logic_vector(7 downto 0);
            dout_1f5 : out std_logic_vector(7 downto 0);
            dout_1f6 : out std_logic_vector(7 downto 0);
            dout_1f7 : out std_logic_vector(7 downto 0);
            dout_1f8 : out std_logic_vector(7 downto 0);
            dout_1f9 : out std_logic_vector(7 downto 0);
            dout_1fa : out std_logic_vector(7 downto 0);
            dout_1fb : out std_logic_vector(7 downto 0);
            dout_1fc : out std_logic_vector(7 downto 0);
            dout_1fd : out std_logic_vector(7 downto 0);
            dout_1fe : out std_logic_vector(7 downto 0);
            dout_1ff : out std_logic_vector(7 downto 0)
        );
    end component;

    component rom_512x512_inhlif2 is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(8 downto 0);
            dout_00 : out std_logic_vector(7 downto 0);
            dout_01 : out std_logic_vector(7 downto 0);
            dout_02 : out std_logic_vector(7 downto 0);
            dout_03 : out std_logic_vector(7 downto 0);
            dout_04 : out std_logic_vector(7 downto 0);
            dout_05 : out std_logic_vector(7 downto 0);
            dout_06 : out std_logic_vector(7 downto 0);
            dout_07 : out std_logic_vector(7 downto 0);
            dout_08 : out std_logic_vector(7 downto 0);
            dout_09 : out std_logic_vector(7 downto 0);
            dout_0a : out std_logic_vector(7 downto 0);
            dout_0b : out std_logic_vector(7 downto 0);
            dout_0c : out std_logic_vector(7 downto 0);
            dout_0d : out std_logic_vector(7 downto 0);
            dout_0e : out std_logic_vector(7 downto 0);
            dout_0f : out std_logic_vector(7 downto 0);
            dout_10 : out std_logic_vector(7 downto 0);
            dout_11 : out std_logic_vector(7 downto 0);
            dout_12 : out std_logic_vector(7 downto 0);
            dout_13 : out std_logic_vector(7 downto 0);
            dout_14 : out std_logic_vector(7 downto 0);
            dout_15 : out std_logic_vector(7 downto 0);
            dout_16 : out std_logic_vector(7 downto 0);
            dout_17 : out std_logic_vector(7 downto 0);
            dout_18 : out std_logic_vector(7 downto 0);
            dout_19 : out std_logic_vector(7 downto 0);
            dout_1a : out std_logic_vector(7 downto 0);
            dout_1b : out std_logic_vector(7 downto 0);
            dout_1c : out std_logic_vector(7 downto 0);
            dout_1d : out std_logic_vector(7 downto 0);
            dout_1e : out std_logic_vector(7 downto 0);
            dout_1f : out std_logic_vector(7 downto 0);
            dout_20 : out std_logic_vector(7 downto 0);
            dout_21 : out std_logic_vector(7 downto 0);
            dout_22 : out std_logic_vector(7 downto 0);
            dout_23 : out std_logic_vector(7 downto 0);
            dout_24 : out std_logic_vector(7 downto 0);
            dout_25 : out std_logic_vector(7 downto 0);
            dout_26 : out std_logic_vector(7 downto 0);
            dout_27 : out std_logic_vector(7 downto 0);
            dout_28 : out std_logic_vector(7 downto 0);
            dout_29 : out std_logic_vector(7 downto 0);
            dout_2a : out std_logic_vector(7 downto 0);
            dout_2b : out std_logic_vector(7 downto 0);
            dout_2c : out std_logic_vector(7 downto 0);
            dout_2d : out std_logic_vector(7 downto 0);
            dout_2e : out std_logic_vector(7 downto 0);
            dout_2f : out std_logic_vector(7 downto 0);
            dout_30 : out std_logic_vector(7 downto 0);
            dout_31 : out std_logic_vector(7 downto 0);
            dout_32 : out std_logic_vector(7 downto 0);
            dout_33 : out std_logic_vector(7 downto 0);
            dout_34 : out std_logic_vector(7 downto 0);
            dout_35 : out std_logic_vector(7 downto 0);
            dout_36 : out std_logic_vector(7 downto 0);
            dout_37 : out std_logic_vector(7 downto 0);
            dout_38 : out std_logic_vector(7 downto 0);
            dout_39 : out std_logic_vector(7 downto 0);
            dout_3a : out std_logic_vector(7 downto 0);
            dout_3b : out std_logic_vector(7 downto 0);
            dout_3c : out std_logic_vector(7 downto 0);
            dout_3d : out std_logic_vector(7 downto 0);
            dout_3e : out std_logic_vector(7 downto 0);
            dout_3f : out std_logic_vector(7 downto 0);
            dout_40 : out std_logic_vector(7 downto 0);
            dout_41 : out std_logic_vector(7 downto 0);
            dout_42 : out std_logic_vector(7 downto 0);
            dout_43 : out std_logic_vector(7 downto 0);
            dout_44 : out std_logic_vector(7 downto 0);
            dout_45 : out std_logic_vector(7 downto 0);
            dout_46 : out std_logic_vector(7 downto 0);
            dout_47 : out std_logic_vector(7 downto 0);
            dout_48 : out std_logic_vector(7 downto 0);
            dout_49 : out std_logic_vector(7 downto 0);
            dout_4a : out std_logic_vector(7 downto 0);
            dout_4b : out std_logic_vector(7 downto 0);
            dout_4c : out std_logic_vector(7 downto 0);
            dout_4d : out std_logic_vector(7 downto 0);
            dout_4e : out std_logic_vector(7 downto 0);
            dout_4f : out std_logic_vector(7 downto 0);
            dout_50 : out std_logic_vector(7 downto 0);
            dout_51 : out std_logic_vector(7 downto 0);
            dout_52 : out std_logic_vector(7 downto 0);
            dout_53 : out std_logic_vector(7 downto 0);
            dout_54 : out std_logic_vector(7 downto 0);
            dout_55 : out std_logic_vector(7 downto 0);
            dout_56 : out std_logic_vector(7 downto 0);
            dout_57 : out std_logic_vector(7 downto 0);
            dout_58 : out std_logic_vector(7 downto 0);
            dout_59 : out std_logic_vector(7 downto 0);
            dout_5a : out std_logic_vector(7 downto 0);
            dout_5b : out std_logic_vector(7 downto 0);
            dout_5c : out std_logic_vector(7 downto 0);
            dout_5d : out std_logic_vector(7 downto 0);
            dout_5e : out std_logic_vector(7 downto 0);
            dout_5f : out std_logic_vector(7 downto 0);
            dout_60 : out std_logic_vector(7 downto 0);
            dout_61 : out std_logic_vector(7 downto 0);
            dout_62 : out std_logic_vector(7 downto 0);
            dout_63 : out std_logic_vector(7 downto 0);
            dout_64 : out std_logic_vector(7 downto 0);
            dout_65 : out std_logic_vector(7 downto 0);
            dout_66 : out std_logic_vector(7 downto 0);
            dout_67 : out std_logic_vector(7 downto 0);
            dout_68 : out std_logic_vector(7 downto 0);
            dout_69 : out std_logic_vector(7 downto 0);
            dout_6a : out std_logic_vector(7 downto 0);
            dout_6b : out std_logic_vector(7 downto 0);
            dout_6c : out std_logic_vector(7 downto 0);
            dout_6d : out std_logic_vector(7 downto 0);
            dout_6e : out std_logic_vector(7 downto 0);
            dout_6f : out std_logic_vector(7 downto 0);
            dout_70 : out std_logic_vector(7 downto 0);
            dout_71 : out std_logic_vector(7 downto 0);
            dout_72 : out std_logic_vector(7 downto 0);
            dout_73 : out std_logic_vector(7 downto 0);
            dout_74 : out std_logic_vector(7 downto 0);
            dout_75 : out std_logic_vector(7 downto 0);
            dout_76 : out std_logic_vector(7 downto 0);
            dout_77 : out std_logic_vector(7 downto 0);
            dout_78 : out std_logic_vector(7 downto 0);
            dout_79 : out std_logic_vector(7 downto 0);
            dout_7a : out std_logic_vector(7 downto 0);
            dout_7b : out std_logic_vector(7 downto 0);
            dout_7c : out std_logic_vector(7 downto 0);
            dout_7d : out std_logic_vector(7 downto 0);
            dout_7e : out std_logic_vector(7 downto 0);
            dout_7f : out std_logic_vector(7 downto 0);
            dout_80 : out std_logic_vector(7 downto 0);
            dout_81 : out std_logic_vector(7 downto 0);
            dout_82 : out std_logic_vector(7 downto 0);
            dout_83 : out std_logic_vector(7 downto 0);
            dout_84 : out std_logic_vector(7 downto 0);
            dout_85 : out std_logic_vector(7 downto 0);
            dout_86 : out std_logic_vector(7 downto 0);
            dout_87 : out std_logic_vector(7 downto 0);
            dout_88 : out std_logic_vector(7 downto 0);
            dout_89 : out std_logic_vector(7 downto 0);
            dout_8a : out std_logic_vector(7 downto 0);
            dout_8b : out std_logic_vector(7 downto 0);
            dout_8c : out std_logic_vector(7 downto 0);
            dout_8d : out std_logic_vector(7 downto 0);
            dout_8e : out std_logic_vector(7 downto 0);
            dout_8f : out std_logic_vector(7 downto 0);
            dout_90 : out std_logic_vector(7 downto 0);
            dout_91 : out std_logic_vector(7 downto 0);
            dout_92 : out std_logic_vector(7 downto 0);
            dout_93 : out std_logic_vector(7 downto 0);
            dout_94 : out std_logic_vector(7 downto 0);
            dout_95 : out std_logic_vector(7 downto 0);
            dout_96 : out std_logic_vector(7 downto 0);
            dout_97 : out std_logic_vector(7 downto 0);
            dout_98 : out std_logic_vector(7 downto 0);
            dout_99 : out std_logic_vector(7 downto 0);
            dout_9a : out std_logic_vector(7 downto 0);
            dout_9b : out std_logic_vector(7 downto 0);
            dout_9c : out std_logic_vector(7 downto 0);
            dout_9d : out std_logic_vector(7 downto 0);
            dout_9e : out std_logic_vector(7 downto 0);
            dout_9f : out std_logic_vector(7 downto 0);
            dout_a0 : out std_logic_vector(7 downto 0);
            dout_a1 : out std_logic_vector(7 downto 0);
            dout_a2 : out std_logic_vector(7 downto 0);
            dout_a3 : out std_logic_vector(7 downto 0);
            dout_a4 : out std_logic_vector(7 downto 0);
            dout_a5 : out std_logic_vector(7 downto 0);
            dout_a6 : out std_logic_vector(7 downto 0);
            dout_a7 : out std_logic_vector(7 downto 0);
            dout_a8 : out std_logic_vector(7 downto 0);
            dout_a9 : out std_logic_vector(7 downto 0);
            dout_aa : out std_logic_vector(7 downto 0);
            dout_ab : out std_logic_vector(7 downto 0);
            dout_ac : out std_logic_vector(7 downto 0);
            dout_ad : out std_logic_vector(7 downto 0);
            dout_ae : out std_logic_vector(7 downto 0);
            dout_af : out std_logic_vector(7 downto 0);
            dout_b0 : out std_logic_vector(7 downto 0);
            dout_b1 : out std_logic_vector(7 downto 0);
            dout_b2 : out std_logic_vector(7 downto 0);
            dout_b3 : out std_logic_vector(7 downto 0);
            dout_b4 : out std_logic_vector(7 downto 0);
            dout_b5 : out std_logic_vector(7 downto 0);
            dout_b6 : out std_logic_vector(7 downto 0);
            dout_b7 : out std_logic_vector(7 downto 0);
            dout_b8 : out std_logic_vector(7 downto 0);
            dout_b9 : out std_logic_vector(7 downto 0);
            dout_ba : out std_logic_vector(7 downto 0);
            dout_bb : out std_logic_vector(7 downto 0);
            dout_bc : out std_logic_vector(7 downto 0);
            dout_bd : out std_logic_vector(7 downto 0);
            dout_be : out std_logic_vector(7 downto 0);
            dout_bf : out std_logic_vector(7 downto 0);
            dout_c0 : out std_logic_vector(7 downto 0);
            dout_c1 : out std_logic_vector(7 downto 0);
            dout_c2 : out std_logic_vector(7 downto 0);
            dout_c3 : out std_logic_vector(7 downto 0);
            dout_c4 : out std_logic_vector(7 downto 0);
            dout_c5 : out std_logic_vector(7 downto 0);
            dout_c6 : out std_logic_vector(7 downto 0);
            dout_c7 : out std_logic_vector(7 downto 0);
            dout_c8 : out std_logic_vector(7 downto 0);
            dout_c9 : out std_logic_vector(7 downto 0);
            dout_ca : out std_logic_vector(7 downto 0);
            dout_cb : out std_logic_vector(7 downto 0);
            dout_cc : out std_logic_vector(7 downto 0);
            dout_cd : out std_logic_vector(7 downto 0);
            dout_ce : out std_logic_vector(7 downto 0);
            dout_cf : out std_logic_vector(7 downto 0);
            dout_d0 : out std_logic_vector(7 downto 0);
            dout_d1 : out std_logic_vector(7 downto 0);
            dout_d2 : out std_logic_vector(7 downto 0);
            dout_d3 : out std_logic_vector(7 downto 0);
            dout_d4 : out std_logic_vector(7 downto 0);
            dout_d5 : out std_logic_vector(7 downto 0);
            dout_d6 : out std_logic_vector(7 downto 0);
            dout_d7 : out std_logic_vector(7 downto 0);
            dout_d8 : out std_logic_vector(7 downto 0);
            dout_d9 : out std_logic_vector(7 downto 0);
            dout_da : out std_logic_vector(7 downto 0);
            dout_db : out std_logic_vector(7 downto 0);
            dout_dc : out std_logic_vector(7 downto 0);
            dout_dd : out std_logic_vector(7 downto 0);
            dout_de : out std_logic_vector(7 downto 0);
            dout_df : out std_logic_vector(7 downto 0);
            dout_e0 : out std_logic_vector(7 downto 0);
            dout_e1 : out std_logic_vector(7 downto 0);
            dout_e2 : out std_logic_vector(7 downto 0);
            dout_e3 : out std_logic_vector(7 downto 0);
            dout_e4 : out std_logic_vector(7 downto 0);
            dout_e5 : out std_logic_vector(7 downto 0);
            dout_e6 : out std_logic_vector(7 downto 0);
            dout_e7 : out std_logic_vector(7 downto 0);
            dout_e8 : out std_logic_vector(7 downto 0);
            dout_e9 : out std_logic_vector(7 downto 0);
            dout_ea : out std_logic_vector(7 downto 0);
            dout_eb : out std_logic_vector(7 downto 0);
            dout_ec : out std_logic_vector(7 downto 0);
            dout_ed : out std_logic_vector(7 downto 0);
            dout_ee : out std_logic_vector(7 downto 0);
            dout_ef : out std_logic_vector(7 downto 0);
            dout_f0 : out std_logic_vector(7 downto 0);
            dout_f1 : out std_logic_vector(7 downto 0);
            dout_f2 : out std_logic_vector(7 downto 0);
            dout_f3 : out std_logic_vector(7 downto 0);
            dout_f4 : out std_logic_vector(7 downto 0);
            dout_f5 : out std_logic_vector(7 downto 0);
            dout_f6 : out std_logic_vector(7 downto 0);
            dout_f7 : out std_logic_vector(7 downto 0);
            dout_f8 : out std_logic_vector(7 downto 0);
            dout_f9 : out std_logic_vector(7 downto 0);
            dout_fa : out std_logic_vector(7 downto 0);
            dout_fb : out std_logic_vector(7 downto 0);
            dout_fc : out std_logic_vector(7 downto 0);
            dout_fd : out std_logic_vector(7 downto 0);
            dout_fe : out std_logic_vector(7 downto 0);
            dout_ff : out std_logic_vector(7 downto 0);
            dout_100 : out std_logic_vector(7 downto 0);
            dout_101 : out std_logic_vector(7 downto 0);
            dout_102 : out std_logic_vector(7 downto 0);
            dout_103 : out std_logic_vector(7 downto 0);
            dout_104 : out std_logic_vector(7 downto 0);
            dout_105 : out std_logic_vector(7 downto 0);
            dout_106 : out std_logic_vector(7 downto 0);
            dout_107 : out std_logic_vector(7 downto 0);
            dout_108 : out std_logic_vector(7 downto 0);
            dout_109 : out std_logic_vector(7 downto 0);
            dout_10a : out std_logic_vector(7 downto 0);
            dout_10b : out std_logic_vector(7 downto 0);
            dout_10c : out std_logic_vector(7 downto 0);
            dout_10d : out std_logic_vector(7 downto 0);
            dout_10e : out std_logic_vector(7 downto 0);
            dout_10f : out std_logic_vector(7 downto 0);
            dout_110 : out std_logic_vector(7 downto 0);
            dout_111 : out std_logic_vector(7 downto 0);
            dout_112 : out std_logic_vector(7 downto 0);
            dout_113 : out std_logic_vector(7 downto 0);
            dout_114 : out std_logic_vector(7 downto 0);
            dout_115 : out std_logic_vector(7 downto 0);
            dout_116 : out std_logic_vector(7 downto 0);
            dout_117 : out std_logic_vector(7 downto 0);
            dout_118 : out std_logic_vector(7 downto 0);
            dout_119 : out std_logic_vector(7 downto 0);
            dout_11a : out std_logic_vector(7 downto 0);
            dout_11b : out std_logic_vector(7 downto 0);
            dout_11c : out std_logic_vector(7 downto 0);
            dout_11d : out std_logic_vector(7 downto 0);
            dout_11e : out std_logic_vector(7 downto 0);
            dout_11f : out std_logic_vector(7 downto 0);
            dout_120 : out std_logic_vector(7 downto 0);
            dout_121 : out std_logic_vector(7 downto 0);
            dout_122 : out std_logic_vector(7 downto 0);
            dout_123 : out std_logic_vector(7 downto 0);
            dout_124 : out std_logic_vector(7 downto 0);
            dout_125 : out std_logic_vector(7 downto 0);
            dout_126 : out std_logic_vector(7 downto 0);
            dout_127 : out std_logic_vector(7 downto 0);
            dout_128 : out std_logic_vector(7 downto 0);
            dout_129 : out std_logic_vector(7 downto 0);
            dout_12a : out std_logic_vector(7 downto 0);
            dout_12b : out std_logic_vector(7 downto 0);
            dout_12c : out std_logic_vector(7 downto 0);
            dout_12d : out std_logic_vector(7 downto 0);
            dout_12e : out std_logic_vector(7 downto 0);
            dout_12f : out std_logic_vector(7 downto 0);
            dout_130 : out std_logic_vector(7 downto 0);
            dout_131 : out std_logic_vector(7 downto 0);
            dout_132 : out std_logic_vector(7 downto 0);
            dout_133 : out std_logic_vector(7 downto 0);
            dout_134 : out std_logic_vector(7 downto 0);
            dout_135 : out std_logic_vector(7 downto 0);
            dout_136 : out std_logic_vector(7 downto 0);
            dout_137 : out std_logic_vector(7 downto 0);
            dout_138 : out std_logic_vector(7 downto 0);
            dout_139 : out std_logic_vector(7 downto 0);
            dout_13a : out std_logic_vector(7 downto 0);
            dout_13b : out std_logic_vector(7 downto 0);
            dout_13c : out std_logic_vector(7 downto 0);
            dout_13d : out std_logic_vector(7 downto 0);
            dout_13e : out std_logic_vector(7 downto 0);
            dout_13f : out std_logic_vector(7 downto 0);
            dout_140 : out std_logic_vector(7 downto 0);
            dout_141 : out std_logic_vector(7 downto 0);
            dout_142 : out std_logic_vector(7 downto 0);
            dout_143 : out std_logic_vector(7 downto 0);
            dout_144 : out std_logic_vector(7 downto 0);
            dout_145 : out std_logic_vector(7 downto 0);
            dout_146 : out std_logic_vector(7 downto 0);
            dout_147 : out std_logic_vector(7 downto 0);
            dout_148 : out std_logic_vector(7 downto 0);
            dout_149 : out std_logic_vector(7 downto 0);
            dout_14a : out std_logic_vector(7 downto 0);
            dout_14b : out std_logic_vector(7 downto 0);
            dout_14c : out std_logic_vector(7 downto 0);
            dout_14d : out std_logic_vector(7 downto 0);
            dout_14e : out std_logic_vector(7 downto 0);
            dout_14f : out std_logic_vector(7 downto 0);
            dout_150 : out std_logic_vector(7 downto 0);
            dout_151 : out std_logic_vector(7 downto 0);
            dout_152 : out std_logic_vector(7 downto 0);
            dout_153 : out std_logic_vector(7 downto 0);
            dout_154 : out std_logic_vector(7 downto 0);
            dout_155 : out std_logic_vector(7 downto 0);
            dout_156 : out std_logic_vector(7 downto 0);
            dout_157 : out std_logic_vector(7 downto 0);
            dout_158 : out std_logic_vector(7 downto 0);
            dout_159 : out std_logic_vector(7 downto 0);
            dout_15a : out std_logic_vector(7 downto 0);
            dout_15b : out std_logic_vector(7 downto 0);
            dout_15c : out std_logic_vector(7 downto 0);
            dout_15d : out std_logic_vector(7 downto 0);
            dout_15e : out std_logic_vector(7 downto 0);
            dout_15f : out std_logic_vector(7 downto 0);
            dout_160 : out std_logic_vector(7 downto 0);
            dout_161 : out std_logic_vector(7 downto 0);
            dout_162 : out std_logic_vector(7 downto 0);
            dout_163 : out std_logic_vector(7 downto 0);
            dout_164 : out std_logic_vector(7 downto 0);
            dout_165 : out std_logic_vector(7 downto 0);
            dout_166 : out std_logic_vector(7 downto 0);
            dout_167 : out std_logic_vector(7 downto 0);
            dout_168 : out std_logic_vector(7 downto 0);
            dout_169 : out std_logic_vector(7 downto 0);
            dout_16a : out std_logic_vector(7 downto 0);
            dout_16b : out std_logic_vector(7 downto 0);
            dout_16c : out std_logic_vector(7 downto 0);
            dout_16d : out std_logic_vector(7 downto 0);
            dout_16e : out std_logic_vector(7 downto 0);
            dout_16f : out std_logic_vector(7 downto 0);
            dout_170 : out std_logic_vector(7 downto 0);
            dout_171 : out std_logic_vector(7 downto 0);
            dout_172 : out std_logic_vector(7 downto 0);
            dout_173 : out std_logic_vector(7 downto 0);
            dout_174 : out std_logic_vector(7 downto 0);
            dout_175 : out std_logic_vector(7 downto 0);
            dout_176 : out std_logic_vector(7 downto 0);
            dout_177 : out std_logic_vector(7 downto 0);
            dout_178 : out std_logic_vector(7 downto 0);
            dout_179 : out std_logic_vector(7 downto 0);
            dout_17a : out std_logic_vector(7 downto 0);
            dout_17b : out std_logic_vector(7 downto 0);
            dout_17c : out std_logic_vector(7 downto 0);
            dout_17d : out std_logic_vector(7 downto 0);
            dout_17e : out std_logic_vector(7 downto 0);
            dout_17f : out std_logic_vector(7 downto 0);
            dout_180 : out std_logic_vector(7 downto 0);
            dout_181 : out std_logic_vector(7 downto 0);
            dout_182 : out std_logic_vector(7 downto 0);
            dout_183 : out std_logic_vector(7 downto 0);
            dout_184 : out std_logic_vector(7 downto 0);
            dout_185 : out std_logic_vector(7 downto 0);
            dout_186 : out std_logic_vector(7 downto 0);
            dout_187 : out std_logic_vector(7 downto 0);
            dout_188 : out std_logic_vector(7 downto 0);
            dout_189 : out std_logic_vector(7 downto 0);
            dout_18a : out std_logic_vector(7 downto 0);
            dout_18b : out std_logic_vector(7 downto 0);
            dout_18c : out std_logic_vector(7 downto 0);
            dout_18d : out std_logic_vector(7 downto 0);
            dout_18e : out std_logic_vector(7 downto 0);
            dout_18f : out std_logic_vector(7 downto 0);
            dout_190 : out std_logic_vector(7 downto 0);
            dout_191 : out std_logic_vector(7 downto 0);
            dout_192 : out std_logic_vector(7 downto 0);
            dout_193 : out std_logic_vector(7 downto 0);
            dout_194 : out std_logic_vector(7 downto 0);
            dout_195 : out std_logic_vector(7 downto 0);
            dout_196 : out std_logic_vector(7 downto 0);
            dout_197 : out std_logic_vector(7 downto 0);
            dout_198 : out std_logic_vector(7 downto 0);
            dout_199 : out std_logic_vector(7 downto 0);
            dout_19a : out std_logic_vector(7 downto 0);
            dout_19b : out std_logic_vector(7 downto 0);
            dout_19c : out std_logic_vector(7 downto 0);
            dout_19d : out std_logic_vector(7 downto 0);
            dout_19e : out std_logic_vector(7 downto 0);
            dout_19f : out std_logic_vector(7 downto 0);
            dout_1a0 : out std_logic_vector(7 downto 0);
            dout_1a1 : out std_logic_vector(7 downto 0);
            dout_1a2 : out std_logic_vector(7 downto 0);
            dout_1a3 : out std_logic_vector(7 downto 0);
            dout_1a4 : out std_logic_vector(7 downto 0);
            dout_1a5 : out std_logic_vector(7 downto 0);
            dout_1a6 : out std_logic_vector(7 downto 0);
            dout_1a7 : out std_logic_vector(7 downto 0);
            dout_1a8 : out std_logic_vector(7 downto 0);
            dout_1a9 : out std_logic_vector(7 downto 0);
            dout_1aa : out std_logic_vector(7 downto 0);
            dout_1ab : out std_logic_vector(7 downto 0);
            dout_1ac : out std_logic_vector(7 downto 0);
            dout_1ad : out std_logic_vector(7 downto 0);
            dout_1ae : out std_logic_vector(7 downto 0);
            dout_1af : out std_logic_vector(7 downto 0);
            dout_1b0 : out std_logic_vector(7 downto 0);
            dout_1b1 : out std_logic_vector(7 downto 0);
            dout_1b2 : out std_logic_vector(7 downto 0);
            dout_1b3 : out std_logic_vector(7 downto 0);
            dout_1b4 : out std_logic_vector(7 downto 0);
            dout_1b5 : out std_logic_vector(7 downto 0);
            dout_1b6 : out std_logic_vector(7 downto 0);
            dout_1b7 : out std_logic_vector(7 downto 0);
            dout_1b8 : out std_logic_vector(7 downto 0);
            dout_1b9 : out std_logic_vector(7 downto 0);
            dout_1ba : out std_logic_vector(7 downto 0);
            dout_1bb : out std_logic_vector(7 downto 0);
            dout_1bc : out std_logic_vector(7 downto 0);
            dout_1bd : out std_logic_vector(7 downto 0);
            dout_1be : out std_logic_vector(7 downto 0);
            dout_1bf : out std_logic_vector(7 downto 0);
            dout_1c0 : out std_logic_vector(7 downto 0);
            dout_1c1 : out std_logic_vector(7 downto 0);
            dout_1c2 : out std_logic_vector(7 downto 0);
            dout_1c3 : out std_logic_vector(7 downto 0);
            dout_1c4 : out std_logic_vector(7 downto 0);
            dout_1c5 : out std_logic_vector(7 downto 0);
            dout_1c6 : out std_logic_vector(7 downto 0);
            dout_1c7 : out std_logic_vector(7 downto 0);
            dout_1c8 : out std_logic_vector(7 downto 0);
            dout_1c9 : out std_logic_vector(7 downto 0);
            dout_1ca : out std_logic_vector(7 downto 0);
            dout_1cb : out std_logic_vector(7 downto 0);
            dout_1cc : out std_logic_vector(7 downto 0);
            dout_1cd : out std_logic_vector(7 downto 0);
            dout_1ce : out std_logic_vector(7 downto 0);
            dout_1cf : out std_logic_vector(7 downto 0);
            dout_1d0 : out std_logic_vector(7 downto 0);
            dout_1d1 : out std_logic_vector(7 downto 0);
            dout_1d2 : out std_logic_vector(7 downto 0);
            dout_1d3 : out std_logic_vector(7 downto 0);
            dout_1d4 : out std_logic_vector(7 downto 0);
            dout_1d5 : out std_logic_vector(7 downto 0);
            dout_1d6 : out std_logic_vector(7 downto 0);
            dout_1d7 : out std_logic_vector(7 downto 0);
            dout_1d8 : out std_logic_vector(7 downto 0);
            dout_1d9 : out std_logic_vector(7 downto 0);
            dout_1da : out std_logic_vector(7 downto 0);
            dout_1db : out std_logic_vector(7 downto 0);
            dout_1dc : out std_logic_vector(7 downto 0);
            dout_1dd : out std_logic_vector(7 downto 0);
            dout_1de : out std_logic_vector(7 downto 0);
            dout_1df : out std_logic_vector(7 downto 0);
            dout_1e0 : out std_logic_vector(7 downto 0);
            dout_1e1 : out std_logic_vector(7 downto 0);
            dout_1e2 : out std_logic_vector(7 downto 0);
            dout_1e3 : out std_logic_vector(7 downto 0);
            dout_1e4 : out std_logic_vector(7 downto 0);
            dout_1e5 : out std_logic_vector(7 downto 0);
            dout_1e6 : out std_logic_vector(7 downto 0);
            dout_1e7 : out std_logic_vector(7 downto 0);
            dout_1e8 : out std_logic_vector(7 downto 0);
            dout_1e9 : out std_logic_vector(7 downto 0);
            dout_1ea : out std_logic_vector(7 downto 0);
            dout_1eb : out std_logic_vector(7 downto 0);
            dout_1ec : out std_logic_vector(7 downto 0);
            dout_1ed : out std_logic_vector(7 downto 0);
            dout_1ee : out std_logic_vector(7 downto 0);
            dout_1ef : out std_logic_vector(7 downto 0);
            dout_1f0 : out std_logic_vector(7 downto 0);
            dout_1f1 : out std_logic_vector(7 downto 0);
            dout_1f2 : out std_logic_vector(7 downto 0);
            dout_1f3 : out std_logic_vector(7 downto 0);
            dout_1f4 : out std_logic_vector(7 downto 0);
            dout_1f5 : out std_logic_vector(7 downto 0);
            dout_1f6 : out std_logic_vector(7 downto 0);
            dout_1f7 : out std_logic_vector(7 downto 0);
            dout_1f8 : out std_logic_vector(7 downto 0);
            dout_1f9 : out std_logic_vector(7 downto 0);
            dout_1fa : out std_logic_vector(7 downto 0);
            dout_1fb : out std_logic_vector(7 downto 0);
            dout_1fc : out std_logic_vector(7 downto 0);
            dout_1fd : out std_logic_vector(7 downto 0);
            dout_1fe : out std_logic_vector(7 downto 0);
            dout_1ff : out std_logic_vector(7 downto 0)
        );
    end component;

    component addr_converter is
        generic (
            N : integer := 10
        );
        port (
            addr_in : in std_logic_vector(N-1 downto 0);
            addr_out : out std_logic_vector(N-1 downto 0)
        );
    end component;

    component barrier is
        generic (
            N : integer := 512
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
    signal out_spikes_inst : std_logic_vector(511 downto 0);
    signal out_sample : std_logic;
    signal neuron_ready_00 : std_logic;
    signal inh_weight_00 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_00 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_01 : std_logic;
    signal inh_weight_01 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_01 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_02 : std_logic;
    signal inh_weight_02 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_02 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_03 : std_logic;
    signal inh_weight_03 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_03 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_04 : std_logic;
    signal inh_weight_04 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_04 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_05 : std_logic;
    signal inh_weight_05 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_05 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_06 : std_logic;
    signal inh_weight_06 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_06 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_07 : std_logic;
    signal inh_weight_07 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_07 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_08 : std_logic;
    signal inh_weight_08 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_08 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_09 : std_logic;
    signal inh_weight_09 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_09 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0a : std_logic;
    signal inh_weight_0a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0b : std_logic;
    signal inh_weight_0b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0c : std_logic;
    signal inh_weight_0c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0d : std_logic;
    signal inh_weight_0d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0e : std_logic;
    signal inh_weight_0e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_0f : std_logic;
    signal inh_weight_0f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_0f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10 : std_logic;
    signal inh_weight_10 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11 : std_logic;
    signal inh_weight_11 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12 : std_logic;
    signal inh_weight_12 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13 : std_logic;
    signal inh_weight_13 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14 : std_logic;
    signal inh_weight_14 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15 : std_logic;
    signal inh_weight_15 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16 : std_logic;
    signal inh_weight_16 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17 : std_logic;
    signal inh_weight_17 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18 : std_logic;
    signal inh_weight_18 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19 : std_logic;
    signal inh_weight_19 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a : std_logic;
    signal inh_weight_1a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b : std_logic;
    signal inh_weight_1b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c : std_logic;
    signal inh_weight_1c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d : std_logic;
    signal inh_weight_1d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e : std_logic;
    signal inh_weight_1e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f : std_logic;
    signal inh_weight_1f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_20 : std_logic;
    signal inh_weight_20 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_20 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_21 : std_logic;
    signal inh_weight_21 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_21 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_22 : std_logic;
    signal inh_weight_22 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_22 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_23 : std_logic;
    signal inh_weight_23 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_23 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_24 : std_logic;
    signal inh_weight_24 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_24 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_25 : std_logic;
    signal inh_weight_25 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_25 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_26 : std_logic;
    signal inh_weight_26 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_26 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_27 : std_logic;
    signal inh_weight_27 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_27 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_28 : std_logic;
    signal inh_weight_28 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_28 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_29 : std_logic;
    signal inh_weight_29 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_29 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2a : std_logic;
    signal inh_weight_2a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2b : std_logic;
    signal inh_weight_2b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2c : std_logic;
    signal inh_weight_2c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2d : std_logic;
    signal inh_weight_2d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2e : std_logic;
    signal inh_weight_2e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_2f : std_logic;
    signal inh_weight_2f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_2f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_30 : std_logic;
    signal inh_weight_30 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_30 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_31 : std_logic;
    signal inh_weight_31 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_31 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_32 : std_logic;
    signal inh_weight_32 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_32 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_33 : std_logic;
    signal inh_weight_33 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_33 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_34 : std_logic;
    signal inh_weight_34 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_34 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_35 : std_logic;
    signal inh_weight_35 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_35 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_36 : std_logic;
    signal inh_weight_36 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_36 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_37 : std_logic;
    signal inh_weight_37 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_37 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_38 : std_logic;
    signal inh_weight_38 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_38 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_39 : std_logic;
    signal inh_weight_39 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_39 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3a : std_logic;
    signal inh_weight_3a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3b : std_logic;
    signal inh_weight_3b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3c : std_logic;
    signal inh_weight_3c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3d : std_logic;
    signal inh_weight_3d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3e : std_logic;
    signal inh_weight_3e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_3f : std_logic;
    signal inh_weight_3f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_3f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_40 : std_logic;
    signal inh_weight_40 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_40 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_41 : std_logic;
    signal inh_weight_41 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_41 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_42 : std_logic;
    signal inh_weight_42 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_42 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_43 : std_logic;
    signal inh_weight_43 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_43 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_44 : std_logic;
    signal inh_weight_44 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_44 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_45 : std_logic;
    signal inh_weight_45 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_45 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_46 : std_logic;
    signal inh_weight_46 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_46 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_47 : std_logic;
    signal inh_weight_47 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_47 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_48 : std_logic;
    signal inh_weight_48 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_48 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_49 : std_logic;
    signal inh_weight_49 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_49 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4a : std_logic;
    signal inh_weight_4a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4b : std_logic;
    signal inh_weight_4b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4c : std_logic;
    signal inh_weight_4c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4d : std_logic;
    signal inh_weight_4d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4e : std_logic;
    signal inh_weight_4e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_4f : std_logic;
    signal inh_weight_4f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_4f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_50 : std_logic;
    signal inh_weight_50 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_50 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_51 : std_logic;
    signal inh_weight_51 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_51 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_52 : std_logic;
    signal inh_weight_52 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_52 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_53 : std_logic;
    signal inh_weight_53 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_53 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_54 : std_logic;
    signal inh_weight_54 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_54 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_55 : std_logic;
    signal inh_weight_55 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_55 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_56 : std_logic;
    signal inh_weight_56 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_56 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_57 : std_logic;
    signal inh_weight_57 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_57 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_58 : std_logic;
    signal inh_weight_58 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_58 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_59 : std_logic;
    signal inh_weight_59 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_59 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5a : std_logic;
    signal inh_weight_5a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5b : std_logic;
    signal inh_weight_5b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5c : std_logic;
    signal inh_weight_5c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5d : std_logic;
    signal inh_weight_5d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5e : std_logic;
    signal inh_weight_5e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_5f : std_logic;
    signal inh_weight_5f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_5f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_60 : std_logic;
    signal inh_weight_60 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_60 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_61 : std_logic;
    signal inh_weight_61 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_61 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_62 : std_logic;
    signal inh_weight_62 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_62 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_63 : std_logic;
    signal inh_weight_63 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_63 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_64 : std_logic;
    signal inh_weight_64 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_64 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_65 : std_logic;
    signal inh_weight_65 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_65 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_66 : std_logic;
    signal inh_weight_66 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_66 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_67 : std_logic;
    signal inh_weight_67 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_67 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_68 : std_logic;
    signal inh_weight_68 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_68 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_69 : std_logic;
    signal inh_weight_69 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_69 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6a : std_logic;
    signal inh_weight_6a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6b : std_logic;
    signal inh_weight_6b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6c : std_logic;
    signal inh_weight_6c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6d : std_logic;
    signal inh_weight_6d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6e : std_logic;
    signal inh_weight_6e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_6f : std_logic;
    signal inh_weight_6f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_6f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_70 : std_logic;
    signal inh_weight_70 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_70 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_71 : std_logic;
    signal inh_weight_71 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_71 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_72 : std_logic;
    signal inh_weight_72 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_72 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_73 : std_logic;
    signal inh_weight_73 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_73 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_74 : std_logic;
    signal inh_weight_74 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_74 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_75 : std_logic;
    signal inh_weight_75 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_75 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_76 : std_logic;
    signal inh_weight_76 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_76 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_77 : std_logic;
    signal inh_weight_77 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_77 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_78 : std_logic;
    signal inh_weight_78 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_78 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_79 : std_logic;
    signal inh_weight_79 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_79 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7a : std_logic;
    signal inh_weight_7a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7b : std_logic;
    signal inh_weight_7b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7c : std_logic;
    signal inh_weight_7c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7d : std_logic;
    signal inh_weight_7d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7e : std_logic;
    signal inh_weight_7e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_7f : std_logic;
    signal inh_weight_7f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_7f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_80 : std_logic;
    signal inh_weight_80 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_80 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_81 : std_logic;
    signal inh_weight_81 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_81 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_82 : std_logic;
    signal inh_weight_82 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_82 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_83 : std_logic;
    signal inh_weight_83 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_83 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_84 : std_logic;
    signal inh_weight_84 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_84 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_85 : std_logic;
    signal inh_weight_85 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_85 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_86 : std_logic;
    signal inh_weight_86 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_86 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_87 : std_logic;
    signal inh_weight_87 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_87 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_88 : std_logic;
    signal inh_weight_88 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_88 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_89 : std_logic;
    signal inh_weight_89 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_89 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8a : std_logic;
    signal inh_weight_8a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8b : std_logic;
    signal inh_weight_8b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8c : std_logic;
    signal inh_weight_8c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8d : std_logic;
    signal inh_weight_8d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8e : std_logic;
    signal inh_weight_8e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_8f : std_logic;
    signal inh_weight_8f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_8f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_90 : std_logic;
    signal inh_weight_90 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_90 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_91 : std_logic;
    signal inh_weight_91 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_91 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_92 : std_logic;
    signal inh_weight_92 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_92 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_93 : std_logic;
    signal inh_weight_93 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_93 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_94 : std_logic;
    signal inh_weight_94 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_94 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_95 : std_logic;
    signal inh_weight_95 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_95 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_96 : std_logic;
    signal inh_weight_96 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_96 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_97 : std_logic;
    signal inh_weight_97 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_97 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_98 : std_logic;
    signal inh_weight_98 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_98 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_99 : std_logic;
    signal inh_weight_99 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_99 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9a : std_logic;
    signal inh_weight_9a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9b : std_logic;
    signal inh_weight_9b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9c : std_logic;
    signal inh_weight_9c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9d : std_logic;
    signal inh_weight_9d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9e : std_logic;
    signal inh_weight_9e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_9f : std_logic;
    signal inh_weight_9f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_9f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a0 : std_logic;
    signal inh_weight_a0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a1 : std_logic;
    signal inh_weight_a1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a2 : std_logic;
    signal inh_weight_a2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a3 : std_logic;
    signal inh_weight_a3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a4 : std_logic;
    signal inh_weight_a4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a5 : std_logic;
    signal inh_weight_a5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a6 : std_logic;
    signal inh_weight_a6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a7 : std_logic;
    signal inh_weight_a7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a8 : std_logic;
    signal inh_weight_a8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_a9 : std_logic;
    signal inh_weight_a9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_a9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_aa : std_logic;
    signal inh_weight_aa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_aa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ab : std_logic;
    signal inh_weight_ab : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ab : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ac : std_logic;
    signal inh_weight_ac : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ac : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ad : std_logic;
    signal inh_weight_ad : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ad : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ae : std_logic;
    signal inh_weight_ae : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ae : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_af : std_logic;
    signal inh_weight_af : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_af : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b0 : std_logic;
    signal inh_weight_b0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b1 : std_logic;
    signal inh_weight_b1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b2 : std_logic;
    signal inh_weight_b2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b3 : std_logic;
    signal inh_weight_b3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b4 : std_logic;
    signal inh_weight_b4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b5 : std_logic;
    signal inh_weight_b5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b6 : std_logic;
    signal inh_weight_b6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b7 : std_logic;
    signal inh_weight_b7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b8 : std_logic;
    signal inh_weight_b8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_b9 : std_logic;
    signal inh_weight_b9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_b9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ba : std_logic;
    signal inh_weight_ba : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ba : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_bb : std_logic;
    signal inh_weight_bb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_bb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_bc : std_logic;
    signal inh_weight_bc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_bc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_bd : std_logic;
    signal inh_weight_bd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_bd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_be : std_logic;
    signal inh_weight_be : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_be : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_bf : std_logic;
    signal inh_weight_bf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_bf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c0 : std_logic;
    signal inh_weight_c0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c1 : std_logic;
    signal inh_weight_c1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c2 : std_logic;
    signal inh_weight_c2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c3 : std_logic;
    signal inh_weight_c3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c4 : std_logic;
    signal inh_weight_c4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c5 : std_logic;
    signal inh_weight_c5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c6 : std_logic;
    signal inh_weight_c6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c7 : std_logic;
    signal inh_weight_c7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c8 : std_logic;
    signal inh_weight_c8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_c9 : std_logic;
    signal inh_weight_c9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_c9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ca : std_logic;
    signal inh_weight_ca : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ca : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_cb : std_logic;
    signal inh_weight_cb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_cb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_cc : std_logic;
    signal inh_weight_cc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_cc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_cd : std_logic;
    signal inh_weight_cd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_cd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ce : std_logic;
    signal inh_weight_ce : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ce : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_cf : std_logic;
    signal inh_weight_cf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_cf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d0 : std_logic;
    signal inh_weight_d0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d1 : std_logic;
    signal inh_weight_d1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d2 : std_logic;
    signal inh_weight_d2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d3 : std_logic;
    signal inh_weight_d3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d4 : std_logic;
    signal inh_weight_d4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d5 : std_logic;
    signal inh_weight_d5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d6 : std_logic;
    signal inh_weight_d6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d7 : std_logic;
    signal inh_weight_d7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d8 : std_logic;
    signal inh_weight_d8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_d9 : std_logic;
    signal inh_weight_d9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_d9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_da : std_logic;
    signal inh_weight_da : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_da : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_db : std_logic;
    signal inh_weight_db : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_db : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_dc : std_logic;
    signal inh_weight_dc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_dc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_dd : std_logic;
    signal inh_weight_dd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_dd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_de : std_logic;
    signal inh_weight_de : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_de : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_df : std_logic;
    signal inh_weight_df : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_df : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e0 : std_logic;
    signal inh_weight_e0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e1 : std_logic;
    signal inh_weight_e1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e2 : std_logic;
    signal inh_weight_e2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e3 : std_logic;
    signal inh_weight_e3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e4 : std_logic;
    signal inh_weight_e4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e5 : std_logic;
    signal inh_weight_e5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e6 : std_logic;
    signal inh_weight_e6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e7 : std_logic;
    signal inh_weight_e7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e8 : std_logic;
    signal inh_weight_e8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_e9 : std_logic;
    signal inh_weight_e9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_e9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ea : std_logic;
    signal inh_weight_ea : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ea : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_eb : std_logic;
    signal inh_weight_eb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_eb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ec : std_logic;
    signal inh_weight_ec : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ec : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ed : std_logic;
    signal inh_weight_ed : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ed : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ee : std_logic;
    signal inh_weight_ee : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ee : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ef : std_logic;
    signal inh_weight_ef : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ef : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f0 : std_logic;
    signal inh_weight_f0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f1 : std_logic;
    signal inh_weight_f1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f2 : std_logic;
    signal inh_weight_f2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f3 : std_logic;
    signal inh_weight_f3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f4 : std_logic;
    signal inh_weight_f4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f5 : std_logic;
    signal inh_weight_f5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f6 : std_logic;
    signal inh_weight_f6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f7 : std_logic;
    signal inh_weight_f7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f8 : std_logic;
    signal inh_weight_f8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_f9 : std_logic;
    signal inh_weight_f9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_f9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_fa : std_logic;
    signal inh_weight_fa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_fa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_fb : std_logic;
    signal inh_weight_fb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_fb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_fc : std_logic;
    signal inh_weight_fc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_fc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_fd : std_logic;
    signal inh_weight_fd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_fd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_fe : std_logic;
    signal inh_weight_fe : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_fe : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_ff : std_logic;
    signal inh_weight_ff : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_ff : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_100 : std_logic;
    signal inh_weight_100 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_100 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_101 : std_logic;
    signal inh_weight_101 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_101 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_102 : std_logic;
    signal inh_weight_102 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_102 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_103 : std_logic;
    signal inh_weight_103 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_103 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_104 : std_logic;
    signal inh_weight_104 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_104 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_105 : std_logic;
    signal inh_weight_105 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_105 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_106 : std_logic;
    signal inh_weight_106 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_106 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_107 : std_logic;
    signal inh_weight_107 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_107 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_108 : std_logic;
    signal inh_weight_108 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_108 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_109 : std_logic;
    signal inh_weight_109 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_109 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10a : std_logic;
    signal inh_weight_10a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10b : std_logic;
    signal inh_weight_10b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10c : std_logic;
    signal inh_weight_10c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10d : std_logic;
    signal inh_weight_10d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10e : std_logic;
    signal inh_weight_10e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_10f : std_logic;
    signal inh_weight_10f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_10f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_110 : std_logic;
    signal inh_weight_110 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_110 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_111 : std_logic;
    signal inh_weight_111 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_111 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_112 : std_logic;
    signal inh_weight_112 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_112 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_113 : std_logic;
    signal inh_weight_113 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_113 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_114 : std_logic;
    signal inh_weight_114 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_114 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_115 : std_logic;
    signal inh_weight_115 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_115 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_116 : std_logic;
    signal inh_weight_116 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_116 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_117 : std_logic;
    signal inh_weight_117 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_117 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_118 : std_logic;
    signal inh_weight_118 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_118 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_119 : std_logic;
    signal inh_weight_119 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_119 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11a : std_logic;
    signal inh_weight_11a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11b : std_logic;
    signal inh_weight_11b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11c : std_logic;
    signal inh_weight_11c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11d : std_logic;
    signal inh_weight_11d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11e : std_logic;
    signal inh_weight_11e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_11f : std_logic;
    signal inh_weight_11f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_11f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_120 : std_logic;
    signal inh_weight_120 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_120 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_121 : std_logic;
    signal inh_weight_121 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_121 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_122 : std_logic;
    signal inh_weight_122 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_122 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_123 : std_logic;
    signal inh_weight_123 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_123 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_124 : std_logic;
    signal inh_weight_124 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_124 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_125 : std_logic;
    signal inh_weight_125 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_125 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_126 : std_logic;
    signal inh_weight_126 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_126 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_127 : std_logic;
    signal inh_weight_127 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_127 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_128 : std_logic;
    signal inh_weight_128 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_128 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_129 : std_logic;
    signal inh_weight_129 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_129 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12a : std_logic;
    signal inh_weight_12a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12b : std_logic;
    signal inh_weight_12b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12c : std_logic;
    signal inh_weight_12c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12d : std_logic;
    signal inh_weight_12d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12e : std_logic;
    signal inh_weight_12e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_12f : std_logic;
    signal inh_weight_12f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_12f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_130 : std_logic;
    signal inh_weight_130 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_130 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_131 : std_logic;
    signal inh_weight_131 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_131 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_132 : std_logic;
    signal inh_weight_132 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_132 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_133 : std_logic;
    signal inh_weight_133 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_133 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_134 : std_logic;
    signal inh_weight_134 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_134 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_135 : std_logic;
    signal inh_weight_135 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_135 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_136 : std_logic;
    signal inh_weight_136 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_136 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_137 : std_logic;
    signal inh_weight_137 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_137 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_138 : std_logic;
    signal inh_weight_138 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_138 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_139 : std_logic;
    signal inh_weight_139 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_139 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13a : std_logic;
    signal inh_weight_13a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13b : std_logic;
    signal inh_weight_13b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13c : std_logic;
    signal inh_weight_13c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13d : std_logic;
    signal inh_weight_13d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13e : std_logic;
    signal inh_weight_13e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_13f : std_logic;
    signal inh_weight_13f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_13f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_140 : std_logic;
    signal inh_weight_140 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_140 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_141 : std_logic;
    signal inh_weight_141 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_141 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_142 : std_logic;
    signal inh_weight_142 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_142 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_143 : std_logic;
    signal inh_weight_143 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_143 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_144 : std_logic;
    signal inh_weight_144 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_144 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_145 : std_logic;
    signal inh_weight_145 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_145 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_146 : std_logic;
    signal inh_weight_146 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_146 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_147 : std_logic;
    signal inh_weight_147 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_147 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_148 : std_logic;
    signal inh_weight_148 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_148 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_149 : std_logic;
    signal inh_weight_149 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_149 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14a : std_logic;
    signal inh_weight_14a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14b : std_logic;
    signal inh_weight_14b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14c : std_logic;
    signal inh_weight_14c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14d : std_logic;
    signal inh_weight_14d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14e : std_logic;
    signal inh_weight_14e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_14f : std_logic;
    signal inh_weight_14f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_14f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_150 : std_logic;
    signal inh_weight_150 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_150 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_151 : std_logic;
    signal inh_weight_151 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_151 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_152 : std_logic;
    signal inh_weight_152 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_152 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_153 : std_logic;
    signal inh_weight_153 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_153 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_154 : std_logic;
    signal inh_weight_154 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_154 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_155 : std_logic;
    signal inh_weight_155 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_155 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_156 : std_logic;
    signal inh_weight_156 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_156 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_157 : std_logic;
    signal inh_weight_157 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_157 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_158 : std_logic;
    signal inh_weight_158 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_158 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_159 : std_logic;
    signal inh_weight_159 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_159 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15a : std_logic;
    signal inh_weight_15a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15b : std_logic;
    signal inh_weight_15b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15c : std_logic;
    signal inh_weight_15c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15d : std_logic;
    signal inh_weight_15d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15e : std_logic;
    signal inh_weight_15e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_15f : std_logic;
    signal inh_weight_15f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_15f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_160 : std_logic;
    signal inh_weight_160 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_160 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_161 : std_logic;
    signal inh_weight_161 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_161 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_162 : std_logic;
    signal inh_weight_162 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_162 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_163 : std_logic;
    signal inh_weight_163 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_163 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_164 : std_logic;
    signal inh_weight_164 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_164 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_165 : std_logic;
    signal inh_weight_165 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_165 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_166 : std_logic;
    signal inh_weight_166 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_166 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_167 : std_logic;
    signal inh_weight_167 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_167 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_168 : std_logic;
    signal inh_weight_168 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_168 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_169 : std_logic;
    signal inh_weight_169 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_169 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16a : std_logic;
    signal inh_weight_16a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16b : std_logic;
    signal inh_weight_16b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16c : std_logic;
    signal inh_weight_16c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16d : std_logic;
    signal inh_weight_16d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16e : std_logic;
    signal inh_weight_16e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_16f : std_logic;
    signal inh_weight_16f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_16f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_170 : std_logic;
    signal inh_weight_170 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_170 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_171 : std_logic;
    signal inh_weight_171 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_171 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_172 : std_logic;
    signal inh_weight_172 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_172 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_173 : std_logic;
    signal inh_weight_173 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_173 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_174 : std_logic;
    signal inh_weight_174 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_174 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_175 : std_logic;
    signal inh_weight_175 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_175 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_176 : std_logic;
    signal inh_weight_176 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_176 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_177 : std_logic;
    signal inh_weight_177 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_177 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_178 : std_logic;
    signal inh_weight_178 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_178 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_179 : std_logic;
    signal inh_weight_179 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_179 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17a : std_logic;
    signal inh_weight_17a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17b : std_logic;
    signal inh_weight_17b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17c : std_logic;
    signal inh_weight_17c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17d : std_logic;
    signal inh_weight_17d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17e : std_logic;
    signal inh_weight_17e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_17f : std_logic;
    signal inh_weight_17f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_17f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_180 : std_logic;
    signal inh_weight_180 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_180 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_181 : std_logic;
    signal inh_weight_181 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_181 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_182 : std_logic;
    signal inh_weight_182 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_182 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_183 : std_logic;
    signal inh_weight_183 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_183 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_184 : std_logic;
    signal inh_weight_184 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_184 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_185 : std_logic;
    signal inh_weight_185 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_185 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_186 : std_logic;
    signal inh_weight_186 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_186 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_187 : std_logic;
    signal inh_weight_187 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_187 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_188 : std_logic;
    signal inh_weight_188 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_188 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_189 : std_logic;
    signal inh_weight_189 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_189 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18a : std_logic;
    signal inh_weight_18a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18b : std_logic;
    signal inh_weight_18b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18c : std_logic;
    signal inh_weight_18c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18d : std_logic;
    signal inh_weight_18d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18e : std_logic;
    signal inh_weight_18e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_18f : std_logic;
    signal inh_weight_18f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_18f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_190 : std_logic;
    signal inh_weight_190 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_190 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_191 : std_logic;
    signal inh_weight_191 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_191 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_192 : std_logic;
    signal inh_weight_192 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_192 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_193 : std_logic;
    signal inh_weight_193 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_193 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_194 : std_logic;
    signal inh_weight_194 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_194 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_195 : std_logic;
    signal inh_weight_195 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_195 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_196 : std_logic;
    signal inh_weight_196 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_196 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_197 : std_logic;
    signal inh_weight_197 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_197 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_198 : std_logic;
    signal inh_weight_198 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_198 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_199 : std_logic;
    signal inh_weight_199 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_199 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19a : std_logic;
    signal inh_weight_19a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19a : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19b : std_logic;
    signal inh_weight_19b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19b : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19c : std_logic;
    signal inh_weight_19c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19c : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19d : std_logic;
    signal inh_weight_19d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19d : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19e : std_logic;
    signal inh_weight_19e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19e : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_19f : std_logic;
    signal inh_weight_19f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_19f : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a0 : std_logic;
    signal inh_weight_1a0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a1 : std_logic;
    signal inh_weight_1a1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a2 : std_logic;
    signal inh_weight_1a2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a3 : std_logic;
    signal inh_weight_1a3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a4 : std_logic;
    signal inh_weight_1a4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a5 : std_logic;
    signal inh_weight_1a5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a6 : std_logic;
    signal inh_weight_1a6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a7 : std_logic;
    signal inh_weight_1a7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a8 : std_logic;
    signal inh_weight_1a8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1a9 : std_logic;
    signal inh_weight_1a9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1a9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1aa : std_logic;
    signal inh_weight_1aa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1aa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ab : std_logic;
    signal inh_weight_1ab : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ab : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ac : std_logic;
    signal inh_weight_1ac : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ac : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ad : std_logic;
    signal inh_weight_1ad : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ad : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ae : std_logic;
    signal inh_weight_1ae : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ae : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1af : std_logic;
    signal inh_weight_1af : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1af : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b0 : std_logic;
    signal inh_weight_1b0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b1 : std_logic;
    signal inh_weight_1b1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b2 : std_logic;
    signal inh_weight_1b2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b3 : std_logic;
    signal inh_weight_1b3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b4 : std_logic;
    signal inh_weight_1b4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b5 : std_logic;
    signal inh_weight_1b5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b6 : std_logic;
    signal inh_weight_1b6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b7 : std_logic;
    signal inh_weight_1b7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b8 : std_logic;
    signal inh_weight_1b8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1b9 : std_logic;
    signal inh_weight_1b9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1b9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ba : std_logic;
    signal inh_weight_1ba : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ba : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1bb : std_logic;
    signal inh_weight_1bb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1bb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1bc : std_logic;
    signal inh_weight_1bc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1bc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1bd : std_logic;
    signal inh_weight_1bd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1bd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1be : std_logic;
    signal inh_weight_1be : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1be : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1bf : std_logic;
    signal inh_weight_1bf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1bf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c0 : std_logic;
    signal inh_weight_1c0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c1 : std_logic;
    signal inh_weight_1c1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c2 : std_logic;
    signal inh_weight_1c2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c3 : std_logic;
    signal inh_weight_1c3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c4 : std_logic;
    signal inh_weight_1c4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c5 : std_logic;
    signal inh_weight_1c5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c6 : std_logic;
    signal inh_weight_1c6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c7 : std_logic;
    signal inh_weight_1c7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c8 : std_logic;
    signal inh_weight_1c8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1c9 : std_logic;
    signal inh_weight_1c9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1c9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ca : std_logic;
    signal inh_weight_1ca : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ca : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1cb : std_logic;
    signal inh_weight_1cb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1cb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1cc : std_logic;
    signal inh_weight_1cc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1cc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1cd : std_logic;
    signal inh_weight_1cd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1cd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ce : std_logic;
    signal inh_weight_1ce : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ce : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1cf : std_logic;
    signal inh_weight_1cf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1cf : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d0 : std_logic;
    signal inh_weight_1d0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d1 : std_logic;
    signal inh_weight_1d1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d2 : std_logic;
    signal inh_weight_1d2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d3 : std_logic;
    signal inh_weight_1d3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d4 : std_logic;
    signal inh_weight_1d4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d5 : std_logic;
    signal inh_weight_1d5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d6 : std_logic;
    signal inh_weight_1d6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d7 : std_logic;
    signal inh_weight_1d7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d8 : std_logic;
    signal inh_weight_1d8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1d9 : std_logic;
    signal inh_weight_1d9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1d9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1da : std_logic;
    signal inh_weight_1da : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1da : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1db : std_logic;
    signal inh_weight_1db : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1db : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1dc : std_logic;
    signal inh_weight_1dc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1dc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1dd : std_logic;
    signal inh_weight_1dd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1dd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1de : std_logic;
    signal inh_weight_1de : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1de : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1df : std_logic;
    signal inh_weight_1df : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1df : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e0 : std_logic;
    signal inh_weight_1e0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e1 : std_logic;
    signal inh_weight_1e1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e2 : std_logic;
    signal inh_weight_1e2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e3 : std_logic;
    signal inh_weight_1e3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e4 : std_logic;
    signal inh_weight_1e4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e5 : std_logic;
    signal inh_weight_1e5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e6 : std_logic;
    signal inh_weight_1e6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e7 : std_logic;
    signal inh_weight_1e7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e8 : std_logic;
    signal inh_weight_1e8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1e9 : std_logic;
    signal inh_weight_1e9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1e9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ea : std_logic;
    signal inh_weight_1ea : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ea : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1eb : std_logic;
    signal inh_weight_1eb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1eb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ec : std_logic;
    signal inh_weight_1ec : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ec : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ed : std_logic;
    signal inh_weight_1ed : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ed : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ee : std_logic;
    signal inh_weight_1ee : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ee : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ef : std_logic;
    signal inh_weight_1ef : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ef : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f0 : std_logic;
    signal inh_weight_1f0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f0 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f1 : std_logic;
    signal inh_weight_1f1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f1 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f2 : std_logic;
    signal inh_weight_1f2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f2 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f3 : std_logic;
    signal inh_weight_1f3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f3 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f4 : std_logic;
    signal inh_weight_1f4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f4 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f5 : std_logic;
    signal inh_weight_1f5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f5 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f6 : std_logic;
    signal inh_weight_1f6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f6 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f7 : std_logic;
    signal inh_weight_1f7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f7 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f8 : std_logic;
    signal inh_weight_1f8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f8 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1f9 : std_logic;
    signal inh_weight_1f9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1f9 : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1fa : std_logic;
    signal inh_weight_1fa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1fa : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1fb : std_logic;
    signal inh_weight_1fb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1fb : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1fc : std_logic;
    signal inh_weight_1fc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1fc : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1fd : std_logic;
    signal inh_weight_1fd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1fd : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1fe : std_logic;
    signal inh_weight_1fe : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1fe : std_logic_vector(neuron_bit_width-1 downto 0);
    signal neuron_ready_1ff : std_logic;
    signal inh_weight_1ff : std_logic_vector(neuron_bit_width-1 downto 0);
    signal exc_weight_1ff : std_logic_vector(neuron_bit_width-1 downto 0);

begin

    neurons_ready <= neuron_ready_00 and neuron_ready_01 and neuron_ready_02 and neuron_ready_03 and neuron_ready_04 and neuron_ready_05 and neuron_ready_06 and neuron_ready_07 and neuron_ready_08 and neuron_ready_09 and neuron_ready_0a and neuron_ready_0b and neuron_ready_0c and neuron_ready_0d and neuron_ready_0e and neuron_ready_0f and neuron_ready_10 and neuron_ready_11 and neuron_ready_12 and neuron_ready_13 and neuron_ready_14 and neuron_ready_15 and neuron_ready_16 and neuron_ready_17 and neuron_ready_18 and neuron_ready_19 and neuron_ready_1a and neuron_ready_1b and neuron_ready_1c and neuron_ready_1d and neuron_ready_1e and neuron_ready_1f and neuron_ready_20 and neuron_ready_21 and neuron_ready_22 and neuron_ready_23 and neuron_ready_24 and neuron_ready_25 and neuron_ready_26 and neuron_ready_27 and neuron_ready_28 and neuron_ready_29 and neuron_ready_2a and neuron_ready_2b and neuron_ready_2c and neuron_ready_2d and neuron_ready_2e and neuron_ready_2f and neuron_ready_30 and neuron_ready_31 and neuron_ready_32 and neuron_ready_33 and neuron_ready_34 and neuron_ready_35 and neuron_ready_36 and neuron_ready_37 and neuron_ready_38 and neuron_ready_39 and neuron_ready_3a and neuron_ready_3b and neuron_ready_3c and neuron_ready_3d and neuron_ready_3e and neuron_ready_3f and neuron_ready_40 and neuron_ready_41 and neuron_ready_42 and neuron_ready_43 and neuron_ready_44 and neuron_ready_45 and neuron_ready_46 and neuron_ready_47 and neuron_ready_48 and neuron_ready_49 and neuron_ready_4a and neuron_ready_4b and neuron_ready_4c and neuron_ready_4d and neuron_ready_4e and neuron_ready_4f and neuron_ready_50 and neuron_ready_51 and neuron_ready_52 and neuron_ready_53 and neuron_ready_54 and neuron_ready_55 and neuron_ready_56 and neuron_ready_57 and neuron_ready_58 and neuron_ready_59 and neuron_ready_5a and neuron_ready_5b and neuron_ready_5c and neuron_ready_5d and neuron_ready_5e and neuron_ready_5f and neuron_ready_60 and neuron_ready_61 and neuron_ready_62 and neuron_ready_63 and neuron_ready_64 and neuron_ready_65 and neuron_ready_66 and neuron_ready_67 and neuron_ready_68 and neuron_ready_69 and neuron_ready_6a and neuron_ready_6b and neuron_ready_6c and neuron_ready_6d and neuron_ready_6e and neuron_ready_6f and neuron_ready_70 and neuron_ready_71 and neuron_ready_72 and neuron_ready_73 and neuron_ready_74 and neuron_ready_75 and neuron_ready_76 and neuron_ready_77 and neuron_ready_78 and neuron_ready_79 and neuron_ready_7a and neuron_ready_7b and neuron_ready_7c and neuron_ready_7d and neuron_ready_7e and neuron_ready_7f and neuron_ready_80 and neuron_ready_81 and neuron_ready_82 and neuron_ready_83 and neuron_ready_84 and neuron_ready_85 and neuron_ready_86 and neuron_ready_87 and neuron_ready_88 and neuron_ready_89 and neuron_ready_8a and neuron_ready_8b and neuron_ready_8c and neuron_ready_8d and neuron_ready_8e and neuron_ready_8f and neuron_ready_90 and neuron_ready_91 and neuron_ready_92 and neuron_ready_93 and neuron_ready_94 and neuron_ready_95 and neuron_ready_96 and neuron_ready_97 and neuron_ready_98 and neuron_ready_99 and neuron_ready_9a and neuron_ready_9b and neuron_ready_9c and neuron_ready_9d and neuron_ready_9e and neuron_ready_9f and neuron_ready_a0 and neuron_ready_a1 and neuron_ready_a2 and neuron_ready_a3 and neuron_ready_a4 and neuron_ready_a5 and neuron_ready_a6 and neuron_ready_a7 and neuron_ready_a8 and neuron_ready_a9 and neuron_ready_aa and neuron_ready_ab and neuron_ready_ac and neuron_ready_ad and neuron_ready_ae and neuron_ready_af and neuron_ready_b0 and neuron_ready_b1 and neuron_ready_b2 and neuron_ready_b3 and neuron_ready_b4 and neuron_ready_b5 and neuron_ready_b6 and neuron_ready_b7 and neuron_ready_b8 and neuron_ready_b9 and neuron_ready_ba and neuron_ready_bb and neuron_ready_bc and neuron_ready_bd and neuron_ready_be and neuron_ready_bf and neuron_ready_c0 and neuron_ready_c1 and neuron_ready_c2 and neuron_ready_c3 and neuron_ready_c4 and neuron_ready_c5 and neuron_ready_c6 and neuron_ready_c7 and neuron_ready_c8 and neuron_ready_c9 and neuron_ready_ca and neuron_ready_cb and neuron_ready_cc and neuron_ready_cd and neuron_ready_ce and neuron_ready_cf and neuron_ready_d0 and neuron_ready_d1 and neuron_ready_d2 and neuron_ready_d3 and neuron_ready_d4 and neuron_ready_d5 and neuron_ready_d6 and neuron_ready_d7 and neuron_ready_d8 and neuron_ready_d9 and neuron_ready_da and neuron_ready_db and neuron_ready_dc and neuron_ready_dd and neuron_ready_de and neuron_ready_df and neuron_ready_e0 and neuron_ready_e1 and neuron_ready_e2 and neuron_ready_e3 and neuron_ready_e4 and neuron_ready_e5 and neuron_ready_e6 and neuron_ready_e7 and neuron_ready_e8 and neuron_ready_e9 and neuron_ready_ea and neuron_ready_eb and neuron_ready_ec and neuron_ready_ed and neuron_ready_ee and neuron_ready_ef and neuron_ready_f0 and neuron_ready_f1 and neuron_ready_f2 and neuron_ready_f3 and neuron_ready_f4 and neuron_ready_f5 and neuron_ready_f6 and neuron_ready_f7 and neuron_ready_f8 and neuron_ready_f9 and neuron_ready_fa and neuron_ready_fb and neuron_ready_fc and neuron_ready_fd and neuron_ready_fe and neuron_ready_ff and neuron_ready_100 and neuron_ready_101 and neuron_ready_102 and neuron_ready_103 and neuron_ready_104 and neuron_ready_105 and neuron_ready_106 and neuron_ready_107 and neuron_ready_108 and neuron_ready_109 and neuron_ready_10a and neuron_ready_10b and neuron_ready_10c and neuron_ready_10d and neuron_ready_10e and neuron_ready_10f and neuron_ready_110 and neuron_ready_111 and neuron_ready_112 and neuron_ready_113 and neuron_ready_114 and neuron_ready_115 and neuron_ready_116 and neuron_ready_117 and neuron_ready_118 and neuron_ready_119 and neuron_ready_11a and neuron_ready_11b and neuron_ready_11c and neuron_ready_11d and neuron_ready_11e and neuron_ready_11f and neuron_ready_120 and neuron_ready_121 and neuron_ready_122 and neuron_ready_123 and neuron_ready_124 and neuron_ready_125 and neuron_ready_126 and neuron_ready_127 and neuron_ready_128 and neuron_ready_129 and neuron_ready_12a and neuron_ready_12b and neuron_ready_12c and neuron_ready_12d and neuron_ready_12e and neuron_ready_12f and neuron_ready_130 and neuron_ready_131 and neuron_ready_132 and neuron_ready_133 and neuron_ready_134 and neuron_ready_135 and neuron_ready_136 and neuron_ready_137 and neuron_ready_138 and neuron_ready_139 and neuron_ready_13a and neuron_ready_13b and neuron_ready_13c and neuron_ready_13d and neuron_ready_13e and neuron_ready_13f and neuron_ready_140 and neuron_ready_141 and neuron_ready_142 and neuron_ready_143 and neuron_ready_144 and neuron_ready_145 and neuron_ready_146 and neuron_ready_147 and neuron_ready_148 and neuron_ready_149 and neuron_ready_14a and neuron_ready_14b and neuron_ready_14c and neuron_ready_14d and neuron_ready_14e and neuron_ready_14f and neuron_ready_150 and neuron_ready_151 and neuron_ready_152 and neuron_ready_153 and neuron_ready_154 and neuron_ready_155 and neuron_ready_156 and neuron_ready_157 and neuron_ready_158 and neuron_ready_159 and neuron_ready_15a and neuron_ready_15b and neuron_ready_15c and neuron_ready_15d and neuron_ready_15e and neuron_ready_15f and neuron_ready_160 and neuron_ready_161 and neuron_ready_162 and neuron_ready_163 and neuron_ready_164 and neuron_ready_165 and neuron_ready_166 and neuron_ready_167 and neuron_ready_168 and neuron_ready_169 and neuron_ready_16a and neuron_ready_16b and neuron_ready_16c and neuron_ready_16d and neuron_ready_16e and neuron_ready_16f and neuron_ready_170 and neuron_ready_171 and neuron_ready_172 and neuron_ready_173 and neuron_ready_174 and neuron_ready_175 and neuron_ready_176 and neuron_ready_177 and neuron_ready_178 and neuron_ready_179 and neuron_ready_17a and neuron_ready_17b and neuron_ready_17c and neuron_ready_17d and neuron_ready_17e and neuron_ready_17f and neuron_ready_180 and neuron_ready_181 and neuron_ready_182 and neuron_ready_183 and neuron_ready_184 and neuron_ready_185 and neuron_ready_186 and neuron_ready_187 and neuron_ready_188 and neuron_ready_189 and neuron_ready_18a and neuron_ready_18b and neuron_ready_18c and neuron_ready_18d and neuron_ready_18e and neuron_ready_18f and neuron_ready_190 and neuron_ready_191 and neuron_ready_192 and neuron_ready_193 and neuron_ready_194 and neuron_ready_195 and neuron_ready_196 and neuron_ready_197 and neuron_ready_198 and neuron_ready_199 and neuron_ready_19a and neuron_ready_19b and neuron_ready_19c and neuron_ready_19d and neuron_ready_19e and neuron_ready_19f and neuron_ready_1a0 and neuron_ready_1a1 and neuron_ready_1a2 and neuron_ready_1a3 and neuron_ready_1a4 and neuron_ready_1a5 and neuron_ready_1a6 and neuron_ready_1a7 and neuron_ready_1a8 and neuron_ready_1a9 and neuron_ready_1aa and neuron_ready_1ab and neuron_ready_1ac and neuron_ready_1ad and neuron_ready_1ae and neuron_ready_1af and neuron_ready_1b0 and neuron_ready_1b1 and neuron_ready_1b2 and neuron_ready_1b3 and neuron_ready_1b4 and neuron_ready_1b5 and neuron_ready_1b6 and neuron_ready_1b7 and neuron_ready_1b8 and neuron_ready_1b9 and neuron_ready_1ba and neuron_ready_1bb and neuron_ready_1bc and neuron_ready_1bd and neuron_ready_1be and neuron_ready_1bf and neuron_ready_1c0 and neuron_ready_1c1 and neuron_ready_1c2 and neuron_ready_1c3 and neuron_ready_1c4 and neuron_ready_1c5 and neuron_ready_1c6 and neuron_ready_1c7 and neuron_ready_1c8 and neuron_ready_1c9 and neuron_ready_1ca and neuron_ready_1cb and neuron_ready_1cc and neuron_ready_1cd and neuron_ready_1ce and neuron_ready_1cf and neuron_ready_1d0 and neuron_ready_1d1 and neuron_ready_1d2 and neuron_ready_1d3 and neuron_ready_1d4 and neuron_ready_1d5 and neuron_ready_1d6 and neuron_ready_1d7 and neuron_ready_1d8 and neuron_ready_1d9 and neuron_ready_1da and neuron_ready_1db and neuron_ready_1dc and neuron_ready_1dd and neuron_ready_1de and neuron_ready_1df and neuron_ready_1e0 and neuron_ready_1e1 and neuron_ready_1e2 and neuron_ready_1e3 and neuron_ready_1e4 and neuron_ready_1e5 and neuron_ready_1e6 and neuron_ready_1e7 and neuron_ready_1e8 and neuron_ready_1e9 and neuron_ready_1ea and neuron_ready_1eb and neuron_ready_1ec and neuron_ready_1ed and neuron_ready_1ee and neuron_ready_1ef and neuron_ready_1f0 and neuron_ready_1f1 and neuron_ready_1f2 and neuron_ready_1f3 and neuron_ready_1f4 and neuron_ready_1f5 and neuron_ready_1f6 and neuron_ready_1f7 and neuron_ready_1f8 and neuron_ready_1f9 and neuron_ready_1fa and neuron_ready_1fb and neuron_ready_1fc and neuron_ready_1fd and neuron_ready_1fe and neuron_ready_1ff and barrier_ready;


    multi_input_control : multi_input_1024_exc_512_inh
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

    neuron_00 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_00,
            inh_weight => signed(inh_weight_00),
            exc_weight => signed(exc_weight_00),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_00,
            out_spike => out_spikes_inst(0)
        );

    neuron_01 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_01,
            inh_weight => signed(inh_weight_01),
            exc_weight => signed(exc_weight_01),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_01,
            out_spike => out_spikes_inst(1)
        );

    neuron_02 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_02,
            inh_weight => signed(inh_weight_02),
            exc_weight => signed(exc_weight_02),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_02,
            out_spike => out_spikes_inst(2)
        );

    neuron_03 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_03,
            inh_weight => signed(inh_weight_03),
            exc_weight => signed(exc_weight_03),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_03,
            out_spike => out_spikes_inst(3)
        );

    neuron_04 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_04,
            inh_weight => signed(inh_weight_04),
            exc_weight => signed(exc_weight_04),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_04,
            out_spike => out_spikes_inst(4)
        );

    neuron_05 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_05,
            inh_weight => signed(inh_weight_05),
            exc_weight => signed(exc_weight_05),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_05,
            out_spike => out_spikes_inst(5)
        );

    neuron_06 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_06,
            inh_weight => signed(inh_weight_06),
            exc_weight => signed(exc_weight_06),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_06,
            out_spike => out_spikes_inst(6)
        );

    neuron_07 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_07,
            inh_weight => signed(inh_weight_07),
            exc_weight => signed(exc_weight_07),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_07,
            out_spike => out_spikes_inst(7)
        );

    neuron_08 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_08,
            inh_weight => signed(inh_weight_08),
            exc_weight => signed(exc_weight_08),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_08,
            out_spike => out_spikes_inst(8)
        );

    neuron_09 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_09,
            inh_weight => signed(inh_weight_09),
            exc_weight => signed(exc_weight_09),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_09,
            out_spike => out_spikes_inst(9)
        );

    neuron_0a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0a,
            inh_weight => signed(inh_weight_0a),
            exc_weight => signed(exc_weight_0a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0a,
            out_spike => out_spikes_inst(10)
        );

    neuron_0b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0b,
            inh_weight => signed(inh_weight_0b),
            exc_weight => signed(exc_weight_0b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0b,
            out_spike => out_spikes_inst(11)
        );

    neuron_0c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0c,
            inh_weight => signed(inh_weight_0c),
            exc_weight => signed(exc_weight_0c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0c,
            out_spike => out_spikes_inst(12)
        );

    neuron_0d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0d,
            inh_weight => signed(inh_weight_0d),
            exc_weight => signed(exc_weight_0d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0d,
            out_spike => out_spikes_inst(13)
        );

    neuron_0e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0e,
            inh_weight => signed(inh_weight_0e),
            exc_weight => signed(exc_weight_0e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0e,
            out_spike => out_spikes_inst(14)
        );

    neuron_0f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_0f,
            inh_weight => signed(inh_weight_0f),
            exc_weight => signed(exc_weight_0f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_0f,
            out_spike => out_spikes_inst(15)
        );

    neuron_10 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10,
            inh_weight => signed(inh_weight_10),
            exc_weight => signed(exc_weight_10),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10,
            out_spike => out_spikes_inst(16)
        );

    neuron_11 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11,
            inh_weight => signed(inh_weight_11),
            exc_weight => signed(exc_weight_11),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11,
            out_spike => out_spikes_inst(17)
        );

    neuron_12 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12,
            inh_weight => signed(inh_weight_12),
            exc_weight => signed(exc_weight_12),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12,
            out_spike => out_spikes_inst(18)
        );

    neuron_13 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13,
            inh_weight => signed(inh_weight_13),
            exc_weight => signed(exc_weight_13),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13,
            out_spike => out_spikes_inst(19)
        );

    neuron_14 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14,
            inh_weight => signed(inh_weight_14),
            exc_weight => signed(exc_weight_14),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14,
            out_spike => out_spikes_inst(20)
        );

    neuron_15 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15,
            inh_weight => signed(inh_weight_15),
            exc_weight => signed(exc_weight_15),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15,
            out_spike => out_spikes_inst(21)
        );

    neuron_16 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16,
            inh_weight => signed(inh_weight_16),
            exc_weight => signed(exc_weight_16),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16,
            out_spike => out_spikes_inst(22)
        );

    neuron_17 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17,
            inh_weight => signed(inh_weight_17),
            exc_weight => signed(exc_weight_17),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17,
            out_spike => out_spikes_inst(23)
        );

    neuron_18 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18,
            inh_weight => signed(inh_weight_18),
            exc_weight => signed(exc_weight_18),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18,
            out_spike => out_spikes_inst(24)
        );

    neuron_19 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19,
            inh_weight => signed(inh_weight_19),
            exc_weight => signed(exc_weight_19),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19,
            out_spike => out_spikes_inst(25)
        );

    neuron_1a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a,
            inh_weight => signed(inh_weight_1a),
            exc_weight => signed(exc_weight_1a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a,
            out_spike => out_spikes_inst(26)
        );

    neuron_1b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b,
            inh_weight => signed(inh_weight_1b),
            exc_weight => signed(exc_weight_1b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b,
            out_spike => out_spikes_inst(27)
        );

    neuron_1c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c,
            inh_weight => signed(inh_weight_1c),
            exc_weight => signed(exc_weight_1c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c,
            out_spike => out_spikes_inst(28)
        );

    neuron_1d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d,
            inh_weight => signed(inh_weight_1d),
            exc_weight => signed(exc_weight_1d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d,
            out_spike => out_spikes_inst(29)
        );

    neuron_1e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e,
            inh_weight => signed(inh_weight_1e),
            exc_weight => signed(exc_weight_1e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e,
            out_spike => out_spikes_inst(30)
        );

    neuron_1f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f,
            inh_weight => signed(inh_weight_1f),
            exc_weight => signed(exc_weight_1f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f,
            out_spike => out_spikes_inst(31)
        );

    neuron_20 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_20,
            inh_weight => signed(inh_weight_20),
            exc_weight => signed(exc_weight_20),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_20,
            out_spike => out_spikes_inst(32)
        );

    neuron_21 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_21,
            inh_weight => signed(inh_weight_21),
            exc_weight => signed(exc_weight_21),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_21,
            out_spike => out_spikes_inst(33)
        );

    neuron_22 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_22,
            inh_weight => signed(inh_weight_22),
            exc_weight => signed(exc_weight_22),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_22,
            out_spike => out_spikes_inst(34)
        );

    neuron_23 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_23,
            inh_weight => signed(inh_weight_23),
            exc_weight => signed(exc_weight_23),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_23,
            out_spike => out_spikes_inst(35)
        );

    neuron_24 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_24,
            inh_weight => signed(inh_weight_24),
            exc_weight => signed(exc_weight_24),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_24,
            out_spike => out_spikes_inst(36)
        );

    neuron_25 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_25,
            inh_weight => signed(inh_weight_25),
            exc_weight => signed(exc_weight_25),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_25,
            out_spike => out_spikes_inst(37)
        );

    neuron_26 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_26,
            inh_weight => signed(inh_weight_26),
            exc_weight => signed(exc_weight_26),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_26,
            out_spike => out_spikes_inst(38)
        );

    neuron_27 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_27,
            inh_weight => signed(inh_weight_27),
            exc_weight => signed(exc_weight_27),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_27,
            out_spike => out_spikes_inst(39)
        );

    neuron_28 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_28,
            inh_weight => signed(inh_weight_28),
            exc_weight => signed(exc_weight_28),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_28,
            out_spike => out_spikes_inst(40)
        );

    neuron_29 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_29,
            inh_weight => signed(inh_weight_29),
            exc_weight => signed(exc_weight_29),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_29,
            out_spike => out_spikes_inst(41)
        );

    neuron_2a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2a,
            inh_weight => signed(inh_weight_2a),
            exc_weight => signed(exc_weight_2a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2a,
            out_spike => out_spikes_inst(42)
        );

    neuron_2b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2b,
            inh_weight => signed(inh_weight_2b),
            exc_weight => signed(exc_weight_2b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2b,
            out_spike => out_spikes_inst(43)
        );

    neuron_2c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2c,
            inh_weight => signed(inh_weight_2c),
            exc_weight => signed(exc_weight_2c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2c,
            out_spike => out_spikes_inst(44)
        );

    neuron_2d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2d,
            inh_weight => signed(inh_weight_2d),
            exc_weight => signed(exc_weight_2d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2d,
            out_spike => out_spikes_inst(45)
        );

    neuron_2e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2e,
            inh_weight => signed(inh_weight_2e),
            exc_weight => signed(exc_weight_2e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2e,
            out_spike => out_spikes_inst(46)
        );

    neuron_2f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_2f,
            inh_weight => signed(inh_weight_2f),
            exc_weight => signed(exc_weight_2f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_2f,
            out_spike => out_spikes_inst(47)
        );

    neuron_30 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_30,
            inh_weight => signed(inh_weight_30),
            exc_weight => signed(exc_weight_30),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_30,
            out_spike => out_spikes_inst(48)
        );

    neuron_31 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_31,
            inh_weight => signed(inh_weight_31),
            exc_weight => signed(exc_weight_31),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_31,
            out_spike => out_spikes_inst(49)
        );

    neuron_32 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_32,
            inh_weight => signed(inh_weight_32),
            exc_weight => signed(exc_weight_32),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_32,
            out_spike => out_spikes_inst(50)
        );

    neuron_33 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_33,
            inh_weight => signed(inh_weight_33),
            exc_weight => signed(exc_weight_33),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_33,
            out_spike => out_spikes_inst(51)
        );

    neuron_34 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_34,
            inh_weight => signed(inh_weight_34),
            exc_weight => signed(exc_weight_34),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_34,
            out_spike => out_spikes_inst(52)
        );

    neuron_35 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_35,
            inh_weight => signed(inh_weight_35),
            exc_weight => signed(exc_weight_35),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_35,
            out_spike => out_spikes_inst(53)
        );

    neuron_36 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_36,
            inh_weight => signed(inh_weight_36),
            exc_weight => signed(exc_weight_36),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_36,
            out_spike => out_spikes_inst(54)
        );

    neuron_37 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_37,
            inh_weight => signed(inh_weight_37),
            exc_weight => signed(exc_weight_37),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_37,
            out_spike => out_spikes_inst(55)
        );

    neuron_38 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_38,
            inh_weight => signed(inh_weight_38),
            exc_weight => signed(exc_weight_38),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_38,
            out_spike => out_spikes_inst(56)
        );

    neuron_39 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_39,
            inh_weight => signed(inh_weight_39),
            exc_weight => signed(exc_weight_39),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_39,
            out_spike => out_spikes_inst(57)
        );

    neuron_3a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3a,
            inh_weight => signed(inh_weight_3a),
            exc_weight => signed(exc_weight_3a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3a,
            out_spike => out_spikes_inst(58)
        );

    neuron_3b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3b,
            inh_weight => signed(inh_weight_3b),
            exc_weight => signed(exc_weight_3b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3b,
            out_spike => out_spikes_inst(59)
        );

    neuron_3c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3c,
            inh_weight => signed(inh_weight_3c),
            exc_weight => signed(exc_weight_3c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3c,
            out_spike => out_spikes_inst(60)
        );

    neuron_3d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3d,
            inh_weight => signed(inh_weight_3d),
            exc_weight => signed(exc_weight_3d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3d,
            out_spike => out_spikes_inst(61)
        );

    neuron_3e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3e,
            inh_weight => signed(inh_weight_3e),
            exc_weight => signed(exc_weight_3e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3e,
            out_spike => out_spikes_inst(62)
        );

    neuron_3f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_3f,
            inh_weight => signed(inh_weight_3f),
            exc_weight => signed(exc_weight_3f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_3f,
            out_spike => out_spikes_inst(63)
        );

    neuron_40 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_40,
            inh_weight => signed(inh_weight_40),
            exc_weight => signed(exc_weight_40),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_40,
            out_spike => out_spikes_inst(64)
        );

    neuron_41 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_41,
            inh_weight => signed(inh_weight_41),
            exc_weight => signed(exc_weight_41),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_41,
            out_spike => out_spikes_inst(65)
        );

    neuron_42 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_42,
            inh_weight => signed(inh_weight_42),
            exc_weight => signed(exc_weight_42),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_42,
            out_spike => out_spikes_inst(66)
        );

    neuron_43 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_43,
            inh_weight => signed(inh_weight_43),
            exc_weight => signed(exc_weight_43),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_43,
            out_spike => out_spikes_inst(67)
        );

    neuron_44 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_44,
            inh_weight => signed(inh_weight_44),
            exc_weight => signed(exc_weight_44),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_44,
            out_spike => out_spikes_inst(68)
        );

    neuron_45 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_45,
            inh_weight => signed(inh_weight_45),
            exc_weight => signed(exc_weight_45),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_45,
            out_spike => out_spikes_inst(69)
        );

    neuron_46 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_46,
            inh_weight => signed(inh_weight_46),
            exc_weight => signed(exc_weight_46),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_46,
            out_spike => out_spikes_inst(70)
        );

    neuron_47 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_47,
            inh_weight => signed(inh_weight_47),
            exc_weight => signed(exc_weight_47),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_47,
            out_spike => out_spikes_inst(71)
        );

    neuron_48 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_48,
            inh_weight => signed(inh_weight_48),
            exc_weight => signed(exc_weight_48),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_48,
            out_spike => out_spikes_inst(72)
        );

    neuron_49 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_49,
            inh_weight => signed(inh_weight_49),
            exc_weight => signed(exc_weight_49),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_49,
            out_spike => out_spikes_inst(73)
        );

    neuron_4a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4a,
            inh_weight => signed(inh_weight_4a),
            exc_weight => signed(exc_weight_4a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4a,
            out_spike => out_spikes_inst(74)
        );

    neuron_4b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4b,
            inh_weight => signed(inh_weight_4b),
            exc_weight => signed(exc_weight_4b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4b,
            out_spike => out_spikes_inst(75)
        );

    neuron_4c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4c,
            inh_weight => signed(inh_weight_4c),
            exc_weight => signed(exc_weight_4c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4c,
            out_spike => out_spikes_inst(76)
        );

    neuron_4d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4d,
            inh_weight => signed(inh_weight_4d),
            exc_weight => signed(exc_weight_4d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4d,
            out_spike => out_spikes_inst(77)
        );

    neuron_4e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4e,
            inh_weight => signed(inh_weight_4e),
            exc_weight => signed(exc_weight_4e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4e,
            out_spike => out_spikes_inst(78)
        );

    neuron_4f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_4f,
            inh_weight => signed(inh_weight_4f),
            exc_weight => signed(exc_weight_4f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_4f,
            out_spike => out_spikes_inst(79)
        );

    neuron_50 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_50,
            inh_weight => signed(inh_weight_50),
            exc_weight => signed(exc_weight_50),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_50,
            out_spike => out_spikes_inst(80)
        );

    neuron_51 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_51,
            inh_weight => signed(inh_weight_51),
            exc_weight => signed(exc_weight_51),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_51,
            out_spike => out_spikes_inst(81)
        );

    neuron_52 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_52,
            inh_weight => signed(inh_weight_52),
            exc_weight => signed(exc_weight_52),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_52,
            out_spike => out_spikes_inst(82)
        );

    neuron_53 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_53,
            inh_weight => signed(inh_weight_53),
            exc_weight => signed(exc_weight_53),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_53,
            out_spike => out_spikes_inst(83)
        );

    neuron_54 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_54,
            inh_weight => signed(inh_weight_54),
            exc_weight => signed(exc_weight_54),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_54,
            out_spike => out_spikes_inst(84)
        );

    neuron_55 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_55,
            inh_weight => signed(inh_weight_55),
            exc_weight => signed(exc_weight_55),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_55,
            out_spike => out_spikes_inst(85)
        );

    neuron_56 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_56,
            inh_weight => signed(inh_weight_56),
            exc_weight => signed(exc_weight_56),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_56,
            out_spike => out_spikes_inst(86)
        );

    neuron_57 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_57,
            inh_weight => signed(inh_weight_57),
            exc_weight => signed(exc_weight_57),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_57,
            out_spike => out_spikes_inst(87)
        );

    neuron_58 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_58,
            inh_weight => signed(inh_weight_58),
            exc_weight => signed(exc_weight_58),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_58,
            out_spike => out_spikes_inst(88)
        );

    neuron_59 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_59,
            inh_weight => signed(inh_weight_59),
            exc_weight => signed(exc_weight_59),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_59,
            out_spike => out_spikes_inst(89)
        );

    neuron_5a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5a,
            inh_weight => signed(inh_weight_5a),
            exc_weight => signed(exc_weight_5a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5a,
            out_spike => out_spikes_inst(90)
        );

    neuron_5b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5b,
            inh_weight => signed(inh_weight_5b),
            exc_weight => signed(exc_weight_5b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5b,
            out_spike => out_spikes_inst(91)
        );

    neuron_5c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5c,
            inh_weight => signed(inh_weight_5c),
            exc_weight => signed(exc_weight_5c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5c,
            out_spike => out_spikes_inst(92)
        );

    neuron_5d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5d,
            inh_weight => signed(inh_weight_5d),
            exc_weight => signed(exc_weight_5d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5d,
            out_spike => out_spikes_inst(93)
        );

    neuron_5e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5e,
            inh_weight => signed(inh_weight_5e),
            exc_weight => signed(exc_weight_5e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5e,
            out_spike => out_spikes_inst(94)
        );

    neuron_5f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_5f,
            inh_weight => signed(inh_weight_5f),
            exc_weight => signed(exc_weight_5f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_5f,
            out_spike => out_spikes_inst(95)
        );

    neuron_60 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_60,
            inh_weight => signed(inh_weight_60),
            exc_weight => signed(exc_weight_60),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_60,
            out_spike => out_spikes_inst(96)
        );

    neuron_61 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_61,
            inh_weight => signed(inh_weight_61),
            exc_weight => signed(exc_weight_61),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_61,
            out_spike => out_spikes_inst(97)
        );

    neuron_62 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_62,
            inh_weight => signed(inh_weight_62),
            exc_weight => signed(exc_weight_62),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_62,
            out_spike => out_spikes_inst(98)
        );

    neuron_63 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_63,
            inh_weight => signed(inh_weight_63),
            exc_weight => signed(exc_weight_63),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_63,
            out_spike => out_spikes_inst(99)
        );

    neuron_64 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_64,
            inh_weight => signed(inh_weight_64),
            exc_weight => signed(exc_weight_64),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_64,
            out_spike => out_spikes_inst(100)
        );

    neuron_65 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_65,
            inh_weight => signed(inh_weight_65),
            exc_weight => signed(exc_weight_65),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_65,
            out_spike => out_spikes_inst(101)
        );

    neuron_66 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_66,
            inh_weight => signed(inh_weight_66),
            exc_weight => signed(exc_weight_66),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_66,
            out_spike => out_spikes_inst(102)
        );

    neuron_67 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_67,
            inh_weight => signed(inh_weight_67),
            exc_weight => signed(exc_weight_67),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_67,
            out_spike => out_spikes_inst(103)
        );

    neuron_68 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_68,
            inh_weight => signed(inh_weight_68),
            exc_weight => signed(exc_weight_68),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_68,
            out_spike => out_spikes_inst(104)
        );

    neuron_69 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_69,
            inh_weight => signed(inh_weight_69),
            exc_weight => signed(exc_weight_69),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_69,
            out_spike => out_spikes_inst(105)
        );

    neuron_6a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6a,
            inh_weight => signed(inh_weight_6a),
            exc_weight => signed(exc_weight_6a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6a,
            out_spike => out_spikes_inst(106)
        );

    neuron_6b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6b,
            inh_weight => signed(inh_weight_6b),
            exc_weight => signed(exc_weight_6b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6b,
            out_spike => out_spikes_inst(107)
        );

    neuron_6c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6c,
            inh_weight => signed(inh_weight_6c),
            exc_weight => signed(exc_weight_6c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6c,
            out_spike => out_spikes_inst(108)
        );

    neuron_6d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6d,
            inh_weight => signed(inh_weight_6d),
            exc_weight => signed(exc_weight_6d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6d,
            out_spike => out_spikes_inst(109)
        );

    neuron_6e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6e,
            inh_weight => signed(inh_weight_6e),
            exc_weight => signed(exc_weight_6e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6e,
            out_spike => out_spikes_inst(110)
        );

    neuron_6f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_6f,
            inh_weight => signed(inh_weight_6f),
            exc_weight => signed(exc_weight_6f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_6f,
            out_spike => out_spikes_inst(111)
        );

    neuron_70 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_70,
            inh_weight => signed(inh_weight_70),
            exc_weight => signed(exc_weight_70),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_70,
            out_spike => out_spikes_inst(112)
        );

    neuron_71 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_71,
            inh_weight => signed(inh_weight_71),
            exc_weight => signed(exc_weight_71),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_71,
            out_spike => out_spikes_inst(113)
        );

    neuron_72 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_72,
            inh_weight => signed(inh_weight_72),
            exc_weight => signed(exc_weight_72),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_72,
            out_spike => out_spikes_inst(114)
        );

    neuron_73 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_73,
            inh_weight => signed(inh_weight_73),
            exc_weight => signed(exc_weight_73),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_73,
            out_spike => out_spikes_inst(115)
        );

    neuron_74 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_74,
            inh_weight => signed(inh_weight_74),
            exc_weight => signed(exc_weight_74),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_74,
            out_spike => out_spikes_inst(116)
        );

    neuron_75 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_75,
            inh_weight => signed(inh_weight_75),
            exc_weight => signed(exc_weight_75),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_75,
            out_spike => out_spikes_inst(117)
        );

    neuron_76 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_76,
            inh_weight => signed(inh_weight_76),
            exc_weight => signed(exc_weight_76),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_76,
            out_spike => out_spikes_inst(118)
        );

    neuron_77 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_77,
            inh_weight => signed(inh_weight_77),
            exc_weight => signed(exc_weight_77),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_77,
            out_spike => out_spikes_inst(119)
        );

    neuron_78 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_78,
            inh_weight => signed(inh_weight_78),
            exc_weight => signed(exc_weight_78),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_78,
            out_spike => out_spikes_inst(120)
        );

    neuron_79 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_79,
            inh_weight => signed(inh_weight_79),
            exc_weight => signed(exc_weight_79),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_79,
            out_spike => out_spikes_inst(121)
        );

    neuron_7a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7a,
            inh_weight => signed(inh_weight_7a),
            exc_weight => signed(exc_weight_7a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7a,
            out_spike => out_spikes_inst(122)
        );

    neuron_7b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7b,
            inh_weight => signed(inh_weight_7b),
            exc_weight => signed(exc_weight_7b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7b,
            out_spike => out_spikes_inst(123)
        );

    neuron_7c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7c,
            inh_weight => signed(inh_weight_7c),
            exc_weight => signed(exc_weight_7c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7c,
            out_spike => out_spikes_inst(124)
        );

    neuron_7d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7d,
            inh_weight => signed(inh_weight_7d),
            exc_weight => signed(exc_weight_7d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7d,
            out_spike => out_spikes_inst(125)
        );

    neuron_7e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7e,
            inh_weight => signed(inh_weight_7e),
            exc_weight => signed(exc_weight_7e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7e,
            out_spike => out_spikes_inst(126)
        );

    neuron_7f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_7f,
            inh_weight => signed(inh_weight_7f),
            exc_weight => signed(exc_weight_7f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_7f,
            out_spike => out_spikes_inst(127)
        );

    neuron_80 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_80,
            inh_weight => signed(inh_weight_80),
            exc_weight => signed(exc_weight_80),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_80,
            out_spike => out_spikes_inst(128)
        );

    neuron_81 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_81,
            inh_weight => signed(inh_weight_81),
            exc_weight => signed(exc_weight_81),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_81,
            out_spike => out_spikes_inst(129)
        );

    neuron_82 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_82,
            inh_weight => signed(inh_weight_82),
            exc_weight => signed(exc_weight_82),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_82,
            out_spike => out_spikes_inst(130)
        );

    neuron_83 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_83,
            inh_weight => signed(inh_weight_83),
            exc_weight => signed(exc_weight_83),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_83,
            out_spike => out_spikes_inst(131)
        );

    neuron_84 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_84,
            inh_weight => signed(inh_weight_84),
            exc_weight => signed(exc_weight_84),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_84,
            out_spike => out_spikes_inst(132)
        );

    neuron_85 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_85,
            inh_weight => signed(inh_weight_85),
            exc_weight => signed(exc_weight_85),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_85,
            out_spike => out_spikes_inst(133)
        );

    neuron_86 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_86,
            inh_weight => signed(inh_weight_86),
            exc_weight => signed(exc_weight_86),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_86,
            out_spike => out_spikes_inst(134)
        );

    neuron_87 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_87,
            inh_weight => signed(inh_weight_87),
            exc_weight => signed(exc_weight_87),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_87,
            out_spike => out_spikes_inst(135)
        );

    neuron_88 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_88,
            inh_weight => signed(inh_weight_88),
            exc_weight => signed(exc_weight_88),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_88,
            out_spike => out_spikes_inst(136)
        );

    neuron_89 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_89,
            inh_weight => signed(inh_weight_89),
            exc_weight => signed(exc_weight_89),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_89,
            out_spike => out_spikes_inst(137)
        );

    neuron_8a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8a,
            inh_weight => signed(inh_weight_8a),
            exc_weight => signed(exc_weight_8a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8a,
            out_spike => out_spikes_inst(138)
        );

    neuron_8b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8b,
            inh_weight => signed(inh_weight_8b),
            exc_weight => signed(exc_weight_8b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8b,
            out_spike => out_spikes_inst(139)
        );

    neuron_8c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8c,
            inh_weight => signed(inh_weight_8c),
            exc_weight => signed(exc_weight_8c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8c,
            out_spike => out_spikes_inst(140)
        );

    neuron_8d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8d,
            inh_weight => signed(inh_weight_8d),
            exc_weight => signed(exc_weight_8d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8d,
            out_spike => out_spikes_inst(141)
        );

    neuron_8e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8e,
            inh_weight => signed(inh_weight_8e),
            exc_weight => signed(exc_weight_8e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8e,
            out_spike => out_spikes_inst(142)
        );

    neuron_8f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_8f,
            inh_weight => signed(inh_weight_8f),
            exc_weight => signed(exc_weight_8f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_8f,
            out_spike => out_spikes_inst(143)
        );

    neuron_90 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_90,
            inh_weight => signed(inh_weight_90),
            exc_weight => signed(exc_weight_90),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_90,
            out_spike => out_spikes_inst(144)
        );

    neuron_91 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_91,
            inh_weight => signed(inh_weight_91),
            exc_weight => signed(exc_weight_91),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_91,
            out_spike => out_spikes_inst(145)
        );

    neuron_92 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_92,
            inh_weight => signed(inh_weight_92),
            exc_weight => signed(exc_weight_92),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_92,
            out_spike => out_spikes_inst(146)
        );

    neuron_93 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_93,
            inh_weight => signed(inh_weight_93),
            exc_weight => signed(exc_weight_93),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_93,
            out_spike => out_spikes_inst(147)
        );

    neuron_94 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_94,
            inh_weight => signed(inh_weight_94),
            exc_weight => signed(exc_weight_94),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_94,
            out_spike => out_spikes_inst(148)
        );

    neuron_95 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_95,
            inh_weight => signed(inh_weight_95),
            exc_weight => signed(exc_weight_95),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_95,
            out_spike => out_spikes_inst(149)
        );

    neuron_96 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_96,
            inh_weight => signed(inh_weight_96),
            exc_weight => signed(exc_weight_96),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_96,
            out_spike => out_spikes_inst(150)
        );

    neuron_97 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_97,
            inh_weight => signed(inh_weight_97),
            exc_weight => signed(exc_weight_97),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_97,
            out_spike => out_spikes_inst(151)
        );

    neuron_98 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_98,
            inh_weight => signed(inh_weight_98),
            exc_weight => signed(exc_weight_98),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_98,
            out_spike => out_spikes_inst(152)
        );

    neuron_99 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_99,
            inh_weight => signed(inh_weight_99),
            exc_weight => signed(exc_weight_99),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_99,
            out_spike => out_spikes_inst(153)
        );

    neuron_9a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9a,
            inh_weight => signed(inh_weight_9a),
            exc_weight => signed(exc_weight_9a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9a,
            out_spike => out_spikes_inst(154)
        );

    neuron_9b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9b,
            inh_weight => signed(inh_weight_9b),
            exc_weight => signed(exc_weight_9b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9b,
            out_spike => out_spikes_inst(155)
        );

    neuron_9c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9c,
            inh_weight => signed(inh_weight_9c),
            exc_weight => signed(exc_weight_9c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9c,
            out_spike => out_spikes_inst(156)
        );

    neuron_9d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9d,
            inh_weight => signed(inh_weight_9d),
            exc_weight => signed(exc_weight_9d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9d,
            out_spike => out_spikes_inst(157)
        );

    neuron_9e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9e,
            inh_weight => signed(inh_weight_9e),
            exc_weight => signed(exc_weight_9e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9e,
            out_spike => out_spikes_inst(158)
        );

    neuron_9f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_9f,
            inh_weight => signed(inh_weight_9f),
            exc_weight => signed(exc_weight_9f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_9f,
            out_spike => out_spikes_inst(159)
        );

    neuron_a0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a0,
            inh_weight => signed(inh_weight_a0),
            exc_weight => signed(exc_weight_a0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a0,
            out_spike => out_spikes_inst(160)
        );

    neuron_a1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a1,
            inh_weight => signed(inh_weight_a1),
            exc_weight => signed(exc_weight_a1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a1,
            out_spike => out_spikes_inst(161)
        );

    neuron_a2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a2,
            inh_weight => signed(inh_weight_a2),
            exc_weight => signed(exc_weight_a2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a2,
            out_spike => out_spikes_inst(162)
        );

    neuron_a3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a3,
            inh_weight => signed(inh_weight_a3),
            exc_weight => signed(exc_weight_a3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a3,
            out_spike => out_spikes_inst(163)
        );

    neuron_a4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a4,
            inh_weight => signed(inh_weight_a4),
            exc_weight => signed(exc_weight_a4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a4,
            out_spike => out_spikes_inst(164)
        );

    neuron_a5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a5,
            inh_weight => signed(inh_weight_a5),
            exc_weight => signed(exc_weight_a5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a5,
            out_spike => out_spikes_inst(165)
        );

    neuron_a6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a6,
            inh_weight => signed(inh_weight_a6),
            exc_weight => signed(exc_weight_a6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a6,
            out_spike => out_spikes_inst(166)
        );

    neuron_a7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a7,
            inh_weight => signed(inh_weight_a7),
            exc_weight => signed(exc_weight_a7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a7,
            out_spike => out_spikes_inst(167)
        );

    neuron_a8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a8,
            inh_weight => signed(inh_weight_a8),
            exc_weight => signed(exc_weight_a8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a8,
            out_spike => out_spikes_inst(168)
        );

    neuron_a9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_a9,
            inh_weight => signed(inh_weight_a9),
            exc_weight => signed(exc_weight_a9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_a9,
            out_spike => out_spikes_inst(169)
        );

    neuron_aa : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_aa,
            inh_weight => signed(inh_weight_aa),
            exc_weight => signed(exc_weight_aa),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_aa,
            out_spike => out_spikes_inst(170)
        );

    neuron_ab : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ab,
            inh_weight => signed(inh_weight_ab),
            exc_weight => signed(exc_weight_ab),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ab,
            out_spike => out_spikes_inst(171)
        );

    neuron_ac : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ac,
            inh_weight => signed(inh_weight_ac),
            exc_weight => signed(exc_weight_ac),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ac,
            out_spike => out_spikes_inst(172)
        );

    neuron_ad : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ad,
            inh_weight => signed(inh_weight_ad),
            exc_weight => signed(exc_weight_ad),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ad,
            out_spike => out_spikes_inst(173)
        );

    neuron_ae : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ae,
            inh_weight => signed(inh_weight_ae),
            exc_weight => signed(exc_weight_ae),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ae,
            out_spike => out_spikes_inst(174)
        );

    neuron_af : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_af,
            inh_weight => signed(inh_weight_af),
            exc_weight => signed(exc_weight_af),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_af,
            out_spike => out_spikes_inst(175)
        );

    neuron_b0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b0,
            inh_weight => signed(inh_weight_b0),
            exc_weight => signed(exc_weight_b0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b0,
            out_spike => out_spikes_inst(176)
        );

    neuron_b1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b1,
            inh_weight => signed(inh_weight_b1),
            exc_weight => signed(exc_weight_b1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b1,
            out_spike => out_spikes_inst(177)
        );

    neuron_b2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b2,
            inh_weight => signed(inh_weight_b2),
            exc_weight => signed(exc_weight_b2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b2,
            out_spike => out_spikes_inst(178)
        );

    neuron_b3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b3,
            inh_weight => signed(inh_weight_b3),
            exc_weight => signed(exc_weight_b3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b3,
            out_spike => out_spikes_inst(179)
        );

    neuron_b4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b4,
            inh_weight => signed(inh_weight_b4),
            exc_weight => signed(exc_weight_b4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b4,
            out_spike => out_spikes_inst(180)
        );

    neuron_b5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b5,
            inh_weight => signed(inh_weight_b5),
            exc_weight => signed(exc_weight_b5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b5,
            out_spike => out_spikes_inst(181)
        );

    neuron_b6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b6,
            inh_weight => signed(inh_weight_b6),
            exc_weight => signed(exc_weight_b6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b6,
            out_spike => out_spikes_inst(182)
        );

    neuron_b7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b7,
            inh_weight => signed(inh_weight_b7),
            exc_weight => signed(exc_weight_b7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b7,
            out_spike => out_spikes_inst(183)
        );

    neuron_b8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b8,
            inh_weight => signed(inh_weight_b8),
            exc_weight => signed(exc_weight_b8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b8,
            out_spike => out_spikes_inst(184)
        );

    neuron_b9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_b9,
            inh_weight => signed(inh_weight_b9),
            exc_weight => signed(exc_weight_b9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_b9,
            out_spike => out_spikes_inst(185)
        );

    neuron_ba : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ba,
            inh_weight => signed(inh_weight_ba),
            exc_weight => signed(exc_weight_ba),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ba,
            out_spike => out_spikes_inst(186)
        );

    neuron_bb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_bb,
            inh_weight => signed(inh_weight_bb),
            exc_weight => signed(exc_weight_bb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_bb,
            out_spike => out_spikes_inst(187)
        );

    neuron_bc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_bc,
            inh_weight => signed(inh_weight_bc),
            exc_weight => signed(exc_weight_bc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_bc,
            out_spike => out_spikes_inst(188)
        );

    neuron_bd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_bd,
            inh_weight => signed(inh_weight_bd),
            exc_weight => signed(exc_weight_bd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_bd,
            out_spike => out_spikes_inst(189)
        );

    neuron_be : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_be,
            inh_weight => signed(inh_weight_be),
            exc_weight => signed(exc_weight_be),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_be,
            out_spike => out_spikes_inst(190)
        );

    neuron_bf : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_bf,
            inh_weight => signed(inh_weight_bf),
            exc_weight => signed(exc_weight_bf),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_bf,
            out_spike => out_spikes_inst(191)
        );

    neuron_c0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c0,
            inh_weight => signed(inh_weight_c0),
            exc_weight => signed(exc_weight_c0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c0,
            out_spike => out_spikes_inst(192)
        );

    neuron_c1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c1,
            inh_weight => signed(inh_weight_c1),
            exc_weight => signed(exc_weight_c1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c1,
            out_spike => out_spikes_inst(193)
        );

    neuron_c2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c2,
            inh_weight => signed(inh_weight_c2),
            exc_weight => signed(exc_weight_c2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c2,
            out_spike => out_spikes_inst(194)
        );

    neuron_c3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c3,
            inh_weight => signed(inh_weight_c3),
            exc_weight => signed(exc_weight_c3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c3,
            out_spike => out_spikes_inst(195)
        );

    neuron_c4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c4,
            inh_weight => signed(inh_weight_c4),
            exc_weight => signed(exc_weight_c4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c4,
            out_spike => out_spikes_inst(196)
        );

    neuron_c5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c5,
            inh_weight => signed(inh_weight_c5),
            exc_weight => signed(exc_weight_c5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c5,
            out_spike => out_spikes_inst(197)
        );

    neuron_c6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c6,
            inh_weight => signed(inh_weight_c6),
            exc_weight => signed(exc_weight_c6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c6,
            out_spike => out_spikes_inst(198)
        );

    neuron_c7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c7,
            inh_weight => signed(inh_weight_c7),
            exc_weight => signed(exc_weight_c7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c7,
            out_spike => out_spikes_inst(199)
        );

    neuron_c8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c8,
            inh_weight => signed(inh_weight_c8),
            exc_weight => signed(exc_weight_c8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c8,
            out_spike => out_spikes_inst(200)
        );

    neuron_c9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_c9,
            inh_weight => signed(inh_weight_c9),
            exc_weight => signed(exc_weight_c9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_c9,
            out_spike => out_spikes_inst(201)
        );

    neuron_ca : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ca,
            inh_weight => signed(inh_weight_ca),
            exc_weight => signed(exc_weight_ca),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ca,
            out_spike => out_spikes_inst(202)
        );

    neuron_cb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_cb,
            inh_weight => signed(inh_weight_cb),
            exc_weight => signed(exc_weight_cb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_cb,
            out_spike => out_spikes_inst(203)
        );

    neuron_cc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_cc,
            inh_weight => signed(inh_weight_cc),
            exc_weight => signed(exc_weight_cc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_cc,
            out_spike => out_spikes_inst(204)
        );

    neuron_cd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_cd,
            inh_weight => signed(inh_weight_cd),
            exc_weight => signed(exc_weight_cd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_cd,
            out_spike => out_spikes_inst(205)
        );

    neuron_ce : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ce,
            inh_weight => signed(inh_weight_ce),
            exc_weight => signed(exc_weight_ce),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ce,
            out_spike => out_spikes_inst(206)
        );

    neuron_cf : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_cf,
            inh_weight => signed(inh_weight_cf),
            exc_weight => signed(exc_weight_cf),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_cf,
            out_spike => out_spikes_inst(207)
        );

    neuron_d0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d0,
            inh_weight => signed(inh_weight_d0),
            exc_weight => signed(exc_weight_d0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d0,
            out_spike => out_spikes_inst(208)
        );

    neuron_d1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d1,
            inh_weight => signed(inh_weight_d1),
            exc_weight => signed(exc_weight_d1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d1,
            out_spike => out_spikes_inst(209)
        );

    neuron_d2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d2,
            inh_weight => signed(inh_weight_d2),
            exc_weight => signed(exc_weight_d2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d2,
            out_spike => out_spikes_inst(210)
        );

    neuron_d3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d3,
            inh_weight => signed(inh_weight_d3),
            exc_weight => signed(exc_weight_d3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d3,
            out_spike => out_spikes_inst(211)
        );

    neuron_d4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d4,
            inh_weight => signed(inh_weight_d4),
            exc_weight => signed(exc_weight_d4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d4,
            out_spike => out_spikes_inst(212)
        );

    neuron_d5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d5,
            inh_weight => signed(inh_weight_d5),
            exc_weight => signed(exc_weight_d5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d5,
            out_spike => out_spikes_inst(213)
        );

    neuron_d6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d6,
            inh_weight => signed(inh_weight_d6),
            exc_weight => signed(exc_weight_d6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d6,
            out_spike => out_spikes_inst(214)
        );

    neuron_d7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d7,
            inh_weight => signed(inh_weight_d7),
            exc_weight => signed(exc_weight_d7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d7,
            out_spike => out_spikes_inst(215)
        );

    neuron_d8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d8,
            inh_weight => signed(inh_weight_d8),
            exc_weight => signed(exc_weight_d8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d8,
            out_spike => out_spikes_inst(216)
        );

    neuron_d9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_d9,
            inh_weight => signed(inh_weight_d9),
            exc_weight => signed(exc_weight_d9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_d9,
            out_spike => out_spikes_inst(217)
        );

    neuron_da : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_da,
            inh_weight => signed(inh_weight_da),
            exc_weight => signed(exc_weight_da),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_da,
            out_spike => out_spikes_inst(218)
        );

    neuron_db : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_db,
            inh_weight => signed(inh_weight_db),
            exc_weight => signed(exc_weight_db),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_db,
            out_spike => out_spikes_inst(219)
        );

    neuron_dc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_dc,
            inh_weight => signed(inh_weight_dc),
            exc_weight => signed(exc_weight_dc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_dc,
            out_spike => out_spikes_inst(220)
        );

    neuron_dd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_dd,
            inh_weight => signed(inh_weight_dd),
            exc_weight => signed(exc_weight_dd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_dd,
            out_spike => out_spikes_inst(221)
        );

    neuron_de : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_de,
            inh_weight => signed(inh_weight_de),
            exc_weight => signed(exc_weight_de),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_de,
            out_spike => out_spikes_inst(222)
        );

    neuron_df : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_df,
            inh_weight => signed(inh_weight_df),
            exc_weight => signed(exc_weight_df),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_df,
            out_spike => out_spikes_inst(223)
        );

    neuron_e0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e0,
            inh_weight => signed(inh_weight_e0),
            exc_weight => signed(exc_weight_e0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e0,
            out_spike => out_spikes_inst(224)
        );

    neuron_e1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e1,
            inh_weight => signed(inh_weight_e1),
            exc_weight => signed(exc_weight_e1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e1,
            out_spike => out_spikes_inst(225)
        );

    neuron_e2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e2,
            inh_weight => signed(inh_weight_e2),
            exc_weight => signed(exc_weight_e2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e2,
            out_spike => out_spikes_inst(226)
        );

    neuron_e3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e3,
            inh_weight => signed(inh_weight_e3),
            exc_weight => signed(exc_weight_e3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e3,
            out_spike => out_spikes_inst(227)
        );

    neuron_e4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e4,
            inh_weight => signed(inh_weight_e4),
            exc_weight => signed(exc_weight_e4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e4,
            out_spike => out_spikes_inst(228)
        );

    neuron_e5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e5,
            inh_weight => signed(inh_weight_e5),
            exc_weight => signed(exc_weight_e5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e5,
            out_spike => out_spikes_inst(229)
        );

    neuron_e6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e6,
            inh_weight => signed(inh_weight_e6),
            exc_weight => signed(exc_weight_e6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e6,
            out_spike => out_spikes_inst(230)
        );

    neuron_e7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e7,
            inh_weight => signed(inh_weight_e7),
            exc_weight => signed(exc_weight_e7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e7,
            out_spike => out_spikes_inst(231)
        );

    neuron_e8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e8,
            inh_weight => signed(inh_weight_e8),
            exc_weight => signed(exc_weight_e8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e8,
            out_spike => out_spikes_inst(232)
        );

    neuron_e9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_e9,
            inh_weight => signed(inh_weight_e9),
            exc_weight => signed(exc_weight_e9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_e9,
            out_spike => out_spikes_inst(233)
        );

    neuron_ea : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ea,
            inh_weight => signed(inh_weight_ea),
            exc_weight => signed(exc_weight_ea),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ea,
            out_spike => out_spikes_inst(234)
        );

    neuron_eb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_eb,
            inh_weight => signed(inh_weight_eb),
            exc_weight => signed(exc_weight_eb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_eb,
            out_spike => out_spikes_inst(235)
        );

    neuron_ec : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ec,
            inh_weight => signed(inh_weight_ec),
            exc_weight => signed(exc_weight_ec),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ec,
            out_spike => out_spikes_inst(236)
        );

    neuron_ed : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ed,
            inh_weight => signed(inh_weight_ed),
            exc_weight => signed(exc_weight_ed),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ed,
            out_spike => out_spikes_inst(237)
        );

    neuron_ee : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ee,
            inh_weight => signed(inh_weight_ee),
            exc_weight => signed(exc_weight_ee),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ee,
            out_spike => out_spikes_inst(238)
        );

    neuron_ef : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ef,
            inh_weight => signed(inh_weight_ef),
            exc_weight => signed(exc_weight_ef),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ef,
            out_spike => out_spikes_inst(239)
        );

    neuron_f0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f0,
            inh_weight => signed(inh_weight_f0),
            exc_weight => signed(exc_weight_f0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f0,
            out_spike => out_spikes_inst(240)
        );

    neuron_f1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f1,
            inh_weight => signed(inh_weight_f1),
            exc_weight => signed(exc_weight_f1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f1,
            out_spike => out_spikes_inst(241)
        );

    neuron_f2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f2,
            inh_weight => signed(inh_weight_f2),
            exc_weight => signed(exc_weight_f2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f2,
            out_spike => out_spikes_inst(242)
        );

    neuron_f3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f3,
            inh_weight => signed(inh_weight_f3),
            exc_weight => signed(exc_weight_f3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f3,
            out_spike => out_spikes_inst(243)
        );

    neuron_f4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f4,
            inh_weight => signed(inh_weight_f4),
            exc_weight => signed(exc_weight_f4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f4,
            out_spike => out_spikes_inst(244)
        );

    neuron_f5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f5,
            inh_weight => signed(inh_weight_f5),
            exc_weight => signed(exc_weight_f5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f5,
            out_spike => out_spikes_inst(245)
        );

    neuron_f6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f6,
            inh_weight => signed(inh_weight_f6),
            exc_weight => signed(exc_weight_f6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f6,
            out_spike => out_spikes_inst(246)
        );

    neuron_f7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f7,
            inh_weight => signed(inh_weight_f7),
            exc_weight => signed(exc_weight_f7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f7,
            out_spike => out_spikes_inst(247)
        );

    neuron_f8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f8,
            inh_weight => signed(inh_weight_f8),
            exc_weight => signed(exc_weight_f8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f8,
            out_spike => out_spikes_inst(248)
        );

    neuron_f9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_f9,
            inh_weight => signed(inh_weight_f9),
            exc_weight => signed(exc_weight_f9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_f9,
            out_spike => out_spikes_inst(249)
        );

    neuron_fa : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_fa,
            inh_weight => signed(inh_weight_fa),
            exc_weight => signed(exc_weight_fa),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_fa,
            out_spike => out_spikes_inst(250)
        );

    neuron_fb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_fb,
            inh_weight => signed(inh_weight_fb),
            exc_weight => signed(exc_weight_fb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_fb,
            out_spike => out_spikes_inst(251)
        );

    neuron_fc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_fc,
            inh_weight => signed(inh_weight_fc),
            exc_weight => signed(exc_weight_fc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_fc,
            out_spike => out_spikes_inst(252)
        );

    neuron_fd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_fd,
            inh_weight => signed(inh_weight_fd),
            exc_weight => signed(exc_weight_fd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_fd,
            out_spike => out_spikes_inst(253)
        );

    neuron_fe : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_fe,
            inh_weight => signed(inh_weight_fe),
            exc_weight => signed(exc_weight_fe),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_fe,
            out_spike => out_spikes_inst(254)
        );

    neuron_ff : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_ff,
            inh_weight => signed(inh_weight_ff),
            exc_weight => signed(exc_weight_ff),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_ff,
            out_spike => out_spikes_inst(255)
        );

    neuron_100 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_100,
            inh_weight => signed(inh_weight_100),
            exc_weight => signed(exc_weight_100),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_100,
            out_spike => out_spikes_inst(256)
        );

    neuron_101 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_101,
            inh_weight => signed(inh_weight_101),
            exc_weight => signed(exc_weight_101),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_101,
            out_spike => out_spikes_inst(257)
        );

    neuron_102 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_102,
            inh_weight => signed(inh_weight_102),
            exc_weight => signed(exc_weight_102),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_102,
            out_spike => out_spikes_inst(258)
        );

    neuron_103 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_103,
            inh_weight => signed(inh_weight_103),
            exc_weight => signed(exc_weight_103),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_103,
            out_spike => out_spikes_inst(259)
        );

    neuron_104 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_104,
            inh_weight => signed(inh_weight_104),
            exc_weight => signed(exc_weight_104),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_104,
            out_spike => out_spikes_inst(260)
        );

    neuron_105 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_105,
            inh_weight => signed(inh_weight_105),
            exc_weight => signed(exc_weight_105),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_105,
            out_spike => out_spikes_inst(261)
        );

    neuron_106 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_106,
            inh_weight => signed(inh_weight_106),
            exc_weight => signed(exc_weight_106),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_106,
            out_spike => out_spikes_inst(262)
        );

    neuron_107 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_107,
            inh_weight => signed(inh_weight_107),
            exc_weight => signed(exc_weight_107),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_107,
            out_spike => out_spikes_inst(263)
        );

    neuron_108 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_108,
            inh_weight => signed(inh_weight_108),
            exc_weight => signed(exc_weight_108),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_108,
            out_spike => out_spikes_inst(264)
        );

    neuron_109 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_109,
            inh_weight => signed(inh_weight_109),
            exc_weight => signed(exc_weight_109),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_109,
            out_spike => out_spikes_inst(265)
        );

    neuron_10a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10a,
            inh_weight => signed(inh_weight_10a),
            exc_weight => signed(exc_weight_10a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10a,
            out_spike => out_spikes_inst(266)
        );

    neuron_10b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10b,
            inh_weight => signed(inh_weight_10b),
            exc_weight => signed(exc_weight_10b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10b,
            out_spike => out_spikes_inst(267)
        );

    neuron_10c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10c,
            inh_weight => signed(inh_weight_10c),
            exc_weight => signed(exc_weight_10c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10c,
            out_spike => out_spikes_inst(268)
        );

    neuron_10d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10d,
            inh_weight => signed(inh_weight_10d),
            exc_weight => signed(exc_weight_10d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10d,
            out_spike => out_spikes_inst(269)
        );

    neuron_10e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10e,
            inh_weight => signed(inh_weight_10e),
            exc_weight => signed(exc_weight_10e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10e,
            out_spike => out_spikes_inst(270)
        );

    neuron_10f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_10f,
            inh_weight => signed(inh_weight_10f),
            exc_weight => signed(exc_weight_10f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_10f,
            out_spike => out_spikes_inst(271)
        );

    neuron_110 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_110,
            inh_weight => signed(inh_weight_110),
            exc_weight => signed(exc_weight_110),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_110,
            out_spike => out_spikes_inst(272)
        );

    neuron_111 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_111,
            inh_weight => signed(inh_weight_111),
            exc_weight => signed(exc_weight_111),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_111,
            out_spike => out_spikes_inst(273)
        );

    neuron_112 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_112,
            inh_weight => signed(inh_weight_112),
            exc_weight => signed(exc_weight_112),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_112,
            out_spike => out_spikes_inst(274)
        );

    neuron_113 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_113,
            inh_weight => signed(inh_weight_113),
            exc_weight => signed(exc_weight_113),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_113,
            out_spike => out_spikes_inst(275)
        );

    neuron_114 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_114,
            inh_weight => signed(inh_weight_114),
            exc_weight => signed(exc_weight_114),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_114,
            out_spike => out_spikes_inst(276)
        );

    neuron_115 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_115,
            inh_weight => signed(inh_weight_115),
            exc_weight => signed(exc_weight_115),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_115,
            out_spike => out_spikes_inst(277)
        );

    neuron_116 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_116,
            inh_weight => signed(inh_weight_116),
            exc_weight => signed(exc_weight_116),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_116,
            out_spike => out_spikes_inst(278)
        );

    neuron_117 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_117,
            inh_weight => signed(inh_weight_117),
            exc_weight => signed(exc_weight_117),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_117,
            out_spike => out_spikes_inst(279)
        );

    neuron_118 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_118,
            inh_weight => signed(inh_weight_118),
            exc_weight => signed(exc_weight_118),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_118,
            out_spike => out_spikes_inst(280)
        );

    neuron_119 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_119,
            inh_weight => signed(inh_weight_119),
            exc_weight => signed(exc_weight_119),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_119,
            out_spike => out_spikes_inst(281)
        );

    neuron_11a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11a,
            inh_weight => signed(inh_weight_11a),
            exc_weight => signed(exc_weight_11a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11a,
            out_spike => out_spikes_inst(282)
        );

    neuron_11b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11b,
            inh_weight => signed(inh_weight_11b),
            exc_weight => signed(exc_weight_11b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11b,
            out_spike => out_spikes_inst(283)
        );

    neuron_11c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11c,
            inh_weight => signed(inh_weight_11c),
            exc_weight => signed(exc_weight_11c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11c,
            out_spike => out_spikes_inst(284)
        );

    neuron_11d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11d,
            inh_weight => signed(inh_weight_11d),
            exc_weight => signed(exc_weight_11d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11d,
            out_spike => out_spikes_inst(285)
        );

    neuron_11e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11e,
            inh_weight => signed(inh_weight_11e),
            exc_weight => signed(exc_weight_11e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11e,
            out_spike => out_spikes_inst(286)
        );

    neuron_11f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_11f,
            inh_weight => signed(inh_weight_11f),
            exc_weight => signed(exc_weight_11f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_11f,
            out_spike => out_spikes_inst(287)
        );

    neuron_120 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_120,
            inh_weight => signed(inh_weight_120),
            exc_weight => signed(exc_weight_120),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_120,
            out_spike => out_spikes_inst(288)
        );

    neuron_121 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_121,
            inh_weight => signed(inh_weight_121),
            exc_weight => signed(exc_weight_121),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_121,
            out_spike => out_spikes_inst(289)
        );

    neuron_122 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_122,
            inh_weight => signed(inh_weight_122),
            exc_weight => signed(exc_weight_122),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_122,
            out_spike => out_spikes_inst(290)
        );

    neuron_123 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_123,
            inh_weight => signed(inh_weight_123),
            exc_weight => signed(exc_weight_123),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_123,
            out_spike => out_spikes_inst(291)
        );

    neuron_124 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_124,
            inh_weight => signed(inh_weight_124),
            exc_weight => signed(exc_weight_124),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_124,
            out_spike => out_spikes_inst(292)
        );

    neuron_125 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_125,
            inh_weight => signed(inh_weight_125),
            exc_weight => signed(exc_weight_125),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_125,
            out_spike => out_spikes_inst(293)
        );

    neuron_126 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_126,
            inh_weight => signed(inh_weight_126),
            exc_weight => signed(exc_weight_126),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_126,
            out_spike => out_spikes_inst(294)
        );

    neuron_127 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_127,
            inh_weight => signed(inh_weight_127),
            exc_weight => signed(exc_weight_127),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_127,
            out_spike => out_spikes_inst(295)
        );

    neuron_128 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_128,
            inh_weight => signed(inh_weight_128),
            exc_weight => signed(exc_weight_128),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_128,
            out_spike => out_spikes_inst(296)
        );

    neuron_129 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_129,
            inh_weight => signed(inh_weight_129),
            exc_weight => signed(exc_weight_129),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_129,
            out_spike => out_spikes_inst(297)
        );

    neuron_12a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12a,
            inh_weight => signed(inh_weight_12a),
            exc_weight => signed(exc_weight_12a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12a,
            out_spike => out_spikes_inst(298)
        );

    neuron_12b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12b,
            inh_weight => signed(inh_weight_12b),
            exc_weight => signed(exc_weight_12b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12b,
            out_spike => out_spikes_inst(299)
        );

    neuron_12c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12c,
            inh_weight => signed(inh_weight_12c),
            exc_weight => signed(exc_weight_12c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12c,
            out_spike => out_spikes_inst(300)
        );

    neuron_12d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12d,
            inh_weight => signed(inh_weight_12d),
            exc_weight => signed(exc_weight_12d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12d,
            out_spike => out_spikes_inst(301)
        );

    neuron_12e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12e,
            inh_weight => signed(inh_weight_12e),
            exc_weight => signed(exc_weight_12e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12e,
            out_spike => out_spikes_inst(302)
        );

    neuron_12f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_12f,
            inh_weight => signed(inh_weight_12f),
            exc_weight => signed(exc_weight_12f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_12f,
            out_spike => out_spikes_inst(303)
        );

    neuron_130 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_130,
            inh_weight => signed(inh_weight_130),
            exc_weight => signed(exc_weight_130),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_130,
            out_spike => out_spikes_inst(304)
        );

    neuron_131 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_131,
            inh_weight => signed(inh_weight_131),
            exc_weight => signed(exc_weight_131),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_131,
            out_spike => out_spikes_inst(305)
        );

    neuron_132 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_132,
            inh_weight => signed(inh_weight_132),
            exc_weight => signed(exc_weight_132),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_132,
            out_spike => out_spikes_inst(306)
        );

    neuron_133 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_133,
            inh_weight => signed(inh_weight_133),
            exc_weight => signed(exc_weight_133),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_133,
            out_spike => out_spikes_inst(307)
        );

    neuron_134 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_134,
            inh_weight => signed(inh_weight_134),
            exc_weight => signed(exc_weight_134),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_134,
            out_spike => out_spikes_inst(308)
        );

    neuron_135 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_135,
            inh_weight => signed(inh_weight_135),
            exc_weight => signed(exc_weight_135),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_135,
            out_spike => out_spikes_inst(309)
        );

    neuron_136 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_136,
            inh_weight => signed(inh_weight_136),
            exc_weight => signed(exc_weight_136),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_136,
            out_spike => out_spikes_inst(310)
        );

    neuron_137 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_137,
            inh_weight => signed(inh_weight_137),
            exc_weight => signed(exc_weight_137),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_137,
            out_spike => out_spikes_inst(311)
        );

    neuron_138 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_138,
            inh_weight => signed(inh_weight_138),
            exc_weight => signed(exc_weight_138),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_138,
            out_spike => out_spikes_inst(312)
        );

    neuron_139 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_139,
            inh_weight => signed(inh_weight_139),
            exc_weight => signed(exc_weight_139),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_139,
            out_spike => out_spikes_inst(313)
        );

    neuron_13a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13a,
            inh_weight => signed(inh_weight_13a),
            exc_weight => signed(exc_weight_13a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13a,
            out_spike => out_spikes_inst(314)
        );

    neuron_13b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13b,
            inh_weight => signed(inh_weight_13b),
            exc_weight => signed(exc_weight_13b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13b,
            out_spike => out_spikes_inst(315)
        );

    neuron_13c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13c,
            inh_weight => signed(inh_weight_13c),
            exc_weight => signed(exc_weight_13c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13c,
            out_spike => out_spikes_inst(316)
        );

    neuron_13d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13d,
            inh_weight => signed(inh_weight_13d),
            exc_weight => signed(exc_weight_13d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13d,
            out_spike => out_spikes_inst(317)
        );

    neuron_13e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13e,
            inh_weight => signed(inh_weight_13e),
            exc_weight => signed(exc_weight_13e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13e,
            out_spike => out_spikes_inst(318)
        );

    neuron_13f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_13f,
            inh_weight => signed(inh_weight_13f),
            exc_weight => signed(exc_weight_13f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_13f,
            out_spike => out_spikes_inst(319)
        );

    neuron_140 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_140,
            inh_weight => signed(inh_weight_140),
            exc_weight => signed(exc_weight_140),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_140,
            out_spike => out_spikes_inst(320)
        );

    neuron_141 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_141,
            inh_weight => signed(inh_weight_141),
            exc_weight => signed(exc_weight_141),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_141,
            out_spike => out_spikes_inst(321)
        );

    neuron_142 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_142,
            inh_weight => signed(inh_weight_142),
            exc_weight => signed(exc_weight_142),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_142,
            out_spike => out_spikes_inst(322)
        );

    neuron_143 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_143,
            inh_weight => signed(inh_weight_143),
            exc_weight => signed(exc_weight_143),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_143,
            out_spike => out_spikes_inst(323)
        );

    neuron_144 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_144,
            inh_weight => signed(inh_weight_144),
            exc_weight => signed(exc_weight_144),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_144,
            out_spike => out_spikes_inst(324)
        );

    neuron_145 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_145,
            inh_weight => signed(inh_weight_145),
            exc_weight => signed(exc_weight_145),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_145,
            out_spike => out_spikes_inst(325)
        );

    neuron_146 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_146,
            inh_weight => signed(inh_weight_146),
            exc_weight => signed(exc_weight_146),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_146,
            out_spike => out_spikes_inst(326)
        );

    neuron_147 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_147,
            inh_weight => signed(inh_weight_147),
            exc_weight => signed(exc_weight_147),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_147,
            out_spike => out_spikes_inst(327)
        );

    neuron_148 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_148,
            inh_weight => signed(inh_weight_148),
            exc_weight => signed(exc_weight_148),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_148,
            out_spike => out_spikes_inst(328)
        );

    neuron_149 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_149,
            inh_weight => signed(inh_weight_149),
            exc_weight => signed(exc_weight_149),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_149,
            out_spike => out_spikes_inst(329)
        );

    neuron_14a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14a,
            inh_weight => signed(inh_weight_14a),
            exc_weight => signed(exc_weight_14a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14a,
            out_spike => out_spikes_inst(330)
        );

    neuron_14b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14b,
            inh_weight => signed(inh_weight_14b),
            exc_weight => signed(exc_weight_14b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14b,
            out_spike => out_spikes_inst(331)
        );

    neuron_14c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14c,
            inh_weight => signed(inh_weight_14c),
            exc_weight => signed(exc_weight_14c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14c,
            out_spike => out_spikes_inst(332)
        );

    neuron_14d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14d,
            inh_weight => signed(inh_weight_14d),
            exc_weight => signed(exc_weight_14d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14d,
            out_spike => out_spikes_inst(333)
        );

    neuron_14e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14e,
            inh_weight => signed(inh_weight_14e),
            exc_weight => signed(exc_weight_14e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14e,
            out_spike => out_spikes_inst(334)
        );

    neuron_14f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_14f,
            inh_weight => signed(inh_weight_14f),
            exc_weight => signed(exc_weight_14f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_14f,
            out_spike => out_spikes_inst(335)
        );

    neuron_150 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_150,
            inh_weight => signed(inh_weight_150),
            exc_weight => signed(exc_weight_150),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_150,
            out_spike => out_spikes_inst(336)
        );

    neuron_151 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_151,
            inh_weight => signed(inh_weight_151),
            exc_weight => signed(exc_weight_151),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_151,
            out_spike => out_spikes_inst(337)
        );

    neuron_152 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_152,
            inh_weight => signed(inh_weight_152),
            exc_weight => signed(exc_weight_152),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_152,
            out_spike => out_spikes_inst(338)
        );

    neuron_153 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_153,
            inh_weight => signed(inh_weight_153),
            exc_weight => signed(exc_weight_153),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_153,
            out_spike => out_spikes_inst(339)
        );

    neuron_154 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_154,
            inh_weight => signed(inh_weight_154),
            exc_weight => signed(exc_weight_154),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_154,
            out_spike => out_spikes_inst(340)
        );

    neuron_155 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_155,
            inh_weight => signed(inh_weight_155),
            exc_weight => signed(exc_weight_155),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_155,
            out_spike => out_spikes_inst(341)
        );

    neuron_156 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_156,
            inh_weight => signed(inh_weight_156),
            exc_weight => signed(exc_weight_156),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_156,
            out_spike => out_spikes_inst(342)
        );

    neuron_157 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_157,
            inh_weight => signed(inh_weight_157),
            exc_weight => signed(exc_weight_157),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_157,
            out_spike => out_spikes_inst(343)
        );

    neuron_158 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_158,
            inh_weight => signed(inh_weight_158),
            exc_weight => signed(exc_weight_158),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_158,
            out_spike => out_spikes_inst(344)
        );

    neuron_159 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_159,
            inh_weight => signed(inh_weight_159),
            exc_weight => signed(exc_weight_159),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_159,
            out_spike => out_spikes_inst(345)
        );

    neuron_15a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15a,
            inh_weight => signed(inh_weight_15a),
            exc_weight => signed(exc_weight_15a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15a,
            out_spike => out_spikes_inst(346)
        );

    neuron_15b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15b,
            inh_weight => signed(inh_weight_15b),
            exc_weight => signed(exc_weight_15b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15b,
            out_spike => out_spikes_inst(347)
        );

    neuron_15c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15c,
            inh_weight => signed(inh_weight_15c),
            exc_weight => signed(exc_weight_15c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15c,
            out_spike => out_spikes_inst(348)
        );

    neuron_15d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15d,
            inh_weight => signed(inh_weight_15d),
            exc_weight => signed(exc_weight_15d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15d,
            out_spike => out_spikes_inst(349)
        );

    neuron_15e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15e,
            inh_weight => signed(inh_weight_15e),
            exc_weight => signed(exc_weight_15e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15e,
            out_spike => out_spikes_inst(350)
        );

    neuron_15f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_15f,
            inh_weight => signed(inh_weight_15f),
            exc_weight => signed(exc_weight_15f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_15f,
            out_spike => out_spikes_inst(351)
        );

    neuron_160 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_160,
            inh_weight => signed(inh_weight_160),
            exc_weight => signed(exc_weight_160),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_160,
            out_spike => out_spikes_inst(352)
        );

    neuron_161 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_161,
            inh_weight => signed(inh_weight_161),
            exc_weight => signed(exc_weight_161),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_161,
            out_spike => out_spikes_inst(353)
        );

    neuron_162 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_162,
            inh_weight => signed(inh_weight_162),
            exc_weight => signed(exc_weight_162),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_162,
            out_spike => out_spikes_inst(354)
        );

    neuron_163 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_163,
            inh_weight => signed(inh_weight_163),
            exc_weight => signed(exc_weight_163),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_163,
            out_spike => out_spikes_inst(355)
        );

    neuron_164 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_164,
            inh_weight => signed(inh_weight_164),
            exc_weight => signed(exc_weight_164),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_164,
            out_spike => out_spikes_inst(356)
        );

    neuron_165 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_165,
            inh_weight => signed(inh_weight_165),
            exc_weight => signed(exc_weight_165),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_165,
            out_spike => out_spikes_inst(357)
        );

    neuron_166 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_166,
            inh_weight => signed(inh_weight_166),
            exc_weight => signed(exc_weight_166),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_166,
            out_spike => out_spikes_inst(358)
        );

    neuron_167 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_167,
            inh_weight => signed(inh_weight_167),
            exc_weight => signed(exc_weight_167),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_167,
            out_spike => out_spikes_inst(359)
        );

    neuron_168 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_168,
            inh_weight => signed(inh_weight_168),
            exc_weight => signed(exc_weight_168),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_168,
            out_spike => out_spikes_inst(360)
        );

    neuron_169 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_169,
            inh_weight => signed(inh_weight_169),
            exc_weight => signed(exc_weight_169),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_169,
            out_spike => out_spikes_inst(361)
        );

    neuron_16a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16a,
            inh_weight => signed(inh_weight_16a),
            exc_weight => signed(exc_weight_16a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16a,
            out_spike => out_spikes_inst(362)
        );

    neuron_16b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16b,
            inh_weight => signed(inh_weight_16b),
            exc_weight => signed(exc_weight_16b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16b,
            out_spike => out_spikes_inst(363)
        );

    neuron_16c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16c,
            inh_weight => signed(inh_weight_16c),
            exc_weight => signed(exc_weight_16c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16c,
            out_spike => out_spikes_inst(364)
        );

    neuron_16d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16d,
            inh_weight => signed(inh_weight_16d),
            exc_weight => signed(exc_weight_16d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16d,
            out_spike => out_spikes_inst(365)
        );

    neuron_16e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16e,
            inh_weight => signed(inh_weight_16e),
            exc_weight => signed(exc_weight_16e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16e,
            out_spike => out_spikes_inst(366)
        );

    neuron_16f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_16f,
            inh_weight => signed(inh_weight_16f),
            exc_weight => signed(exc_weight_16f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_16f,
            out_spike => out_spikes_inst(367)
        );

    neuron_170 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_170,
            inh_weight => signed(inh_weight_170),
            exc_weight => signed(exc_weight_170),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_170,
            out_spike => out_spikes_inst(368)
        );

    neuron_171 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_171,
            inh_weight => signed(inh_weight_171),
            exc_weight => signed(exc_weight_171),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_171,
            out_spike => out_spikes_inst(369)
        );

    neuron_172 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_172,
            inh_weight => signed(inh_weight_172),
            exc_weight => signed(exc_weight_172),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_172,
            out_spike => out_spikes_inst(370)
        );

    neuron_173 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_173,
            inh_weight => signed(inh_weight_173),
            exc_weight => signed(exc_weight_173),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_173,
            out_spike => out_spikes_inst(371)
        );

    neuron_174 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_174,
            inh_weight => signed(inh_weight_174),
            exc_weight => signed(exc_weight_174),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_174,
            out_spike => out_spikes_inst(372)
        );

    neuron_175 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_175,
            inh_weight => signed(inh_weight_175),
            exc_weight => signed(exc_weight_175),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_175,
            out_spike => out_spikes_inst(373)
        );

    neuron_176 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_176,
            inh_weight => signed(inh_weight_176),
            exc_weight => signed(exc_weight_176),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_176,
            out_spike => out_spikes_inst(374)
        );

    neuron_177 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_177,
            inh_weight => signed(inh_weight_177),
            exc_weight => signed(exc_weight_177),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_177,
            out_spike => out_spikes_inst(375)
        );

    neuron_178 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_178,
            inh_weight => signed(inh_weight_178),
            exc_weight => signed(exc_weight_178),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_178,
            out_spike => out_spikes_inst(376)
        );

    neuron_179 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_179,
            inh_weight => signed(inh_weight_179),
            exc_weight => signed(exc_weight_179),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_179,
            out_spike => out_spikes_inst(377)
        );

    neuron_17a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17a,
            inh_weight => signed(inh_weight_17a),
            exc_weight => signed(exc_weight_17a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17a,
            out_spike => out_spikes_inst(378)
        );

    neuron_17b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17b,
            inh_weight => signed(inh_weight_17b),
            exc_weight => signed(exc_weight_17b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17b,
            out_spike => out_spikes_inst(379)
        );

    neuron_17c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17c,
            inh_weight => signed(inh_weight_17c),
            exc_weight => signed(exc_weight_17c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17c,
            out_spike => out_spikes_inst(380)
        );

    neuron_17d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17d,
            inh_weight => signed(inh_weight_17d),
            exc_weight => signed(exc_weight_17d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17d,
            out_spike => out_spikes_inst(381)
        );

    neuron_17e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17e,
            inh_weight => signed(inh_weight_17e),
            exc_weight => signed(exc_weight_17e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17e,
            out_spike => out_spikes_inst(382)
        );

    neuron_17f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_17f,
            inh_weight => signed(inh_weight_17f),
            exc_weight => signed(exc_weight_17f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_17f,
            out_spike => out_spikes_inst(383)
        );

    neuron_180 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_180,
            inh_weight => signed(inh_weight_180),
            exc_weight => signed(exc_weight_180),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_180,
            out_spike => out_spikes_inst(384)
        );

    neuron_181 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_181,
            inh_weight => signed(inh_weight_181),
            exc_weight => signed(exc_weight_181),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_181,
            out_spike => out_spikes_inst(385)
        );

    neuron_182 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_182,
            inh_weight => signed(inh_weight_182),
            exc_weight => signed(exc_weight_182),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_182,
            out_spike => out_spikes_inst(386)
        );

    neuron_183 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_183,
            inh_weight => signed(inh_weight_183),
            exc_weight => signed(exc_weight_183),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_183,
            out_spike => out_spikes_inst(387)
        );

    neuron_184 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_184,
            inh_weight => signed(inh_weight_184),
            exc_weight => signed(exc_weight_184),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_184,
            out_spike => out_spikes_inst(388)
        );

    neuron_185 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_185,
            inh_weight => signed(inh_weight_185),
            exc_weight => signed(exc_weight_185),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_185,
            out_spike => out_spikes_inst(389)
        );

    neuron_186 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_186,
            inh_weight => signed(inh_weight_186),
            exc_weight => signed(exc_weight_186),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_186,
            out_spike => out_spikes_inst(390)
        );

    neuron_187 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_187,
            inh_weight => signed(inh_weight_187),
            exc_weight => signed(exc_weight_187),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_187,
            out_spike => out_spikes_inst(391)
        );

    neuron_188 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_188,
            inh_weight => signed(inh_weight_188),
            exc_weight => signed(exc_weight_188),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_188,
            out_spike => out_spikes_inst(392)
        );

    neuron_189 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_189,
            inh_weight => signed(inh_weight_189),
            exc_weight => signed(exc_weight_189),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_189,
            out_spike => out_spikes_inst(393)
        );

    neuron_18a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18a,
            inh_weight => signed(inh_weight_18a),
            exc_weight => signed(exc_weight_18a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18a,
            out_spike => out_spikes_inst(394)
        );

    neuron_18b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18b,
            inh_weight => signed(inh_weight_18b),
            exc_weight => signed(exc_weight_18b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18b,
            out_spike => out_spikes_inst(395)
        );

    neuron_18c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18c,
            inh_weight => signed(inh_weight_18c),
            exc_weight => signed(exc_weight_18c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18c,
            out_spike => out_spikes_inst(396)
        );

    neuron_18d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18d,
            inh_weight => signed(inh_weight_18d),
            exc_weight => signed(exc_weight_18d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18d,
            out_spike => out_spikes_inst(397)
        );

    neuron_18e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18e,
            inh_weight => signed(inh_weight_18e),
            exc_weight => signed(exc_weight_18e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18e,
            out_spike => out_spikes_inst(398)
        );

    neuron_18f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_18f,
            inh_weight => signed(inh_weight_18f),
            exc_weight => signed(exc_weight_18f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_18f,
            out_spike => out_spikes_inst(399)
        );

    neuron_190 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_190,
            inh_weight => signed(inh_weight_190),
            exc_weight => signed(exc_weight_190),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_190,
            out_spike => out_spikes_inst(400)
        );

    neuron_191 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_191,
            inh_weight => signed(inh_weight_191),
            exc_weight => signed(exc_weight_191),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_191,
            out_spike => out_spikes_inst(401)
        );

    neuron_192 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_192,
            inh_weight => signed(inh_weight_192),
            exc_weight => signed(exc_weight_192),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_192,
            out_spike => out_spikes_inst(402)
        );

    neuron_193 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_193,
            inh_weight => signed(inh_weight_193),
            exc_weight => signed(exc_weight_193),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_193,
            out_spike => out_spikes_inst(403)
        );

    neuron_194 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_194,
            inh_weight => signed(inh_weight_194),
            exc_weight => signed(exc_weight_194),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_194,
            out_spike => out_spikes_inst(404)
        );

    neuron_195 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_195,
            inh_weight => signed(inh_weight_195),
            exc_weight => signed(exc_weight_195),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_195,
            out_spike => out_spikes_inst(405)
        );

    neuron_196 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_196,
            inh_weight => signed(inh_weight_196),
            exc_weight => signed(exc_weight_196),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_196,
            out_spike => out_spikes_inst(406)
        );

    neuron_197 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_197,
            inh_weight => signed(inh_weight_197),
            exc_weight => signed(exc_weight_197),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_197,
            out_spike => out_spikes_inst(407)
        );

    neuron_198 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_198,
            inh_weight => signed(inh_weight_198),
            exc_weight => signed(exc_weight_198),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_198,
            out_spike => out_spikes_inst(408)
        );

    neuron_199 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_199,
            inh_weight => signed(inh_weight_199),
            exc_weight => signed(exc_weight_199),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_199,
            out_spike => out_spikes_inst(409)
        );

    neuron_19a : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19a,
            inh_weight => signed(inh_weight_19a),
            exc_weight => signed(exc_weight_19a),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19a,
            out_spike => out_spikes_inst(410)
        );

    neuron_19b : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19b,
            inh_weight => signed(inh_weight_19b),
            exc_weight => signed(exc_weight_19b),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19b,
            out_spike => out_spikes_inst(411)
        );

    neuron_19c : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19c,
            inh_weight => signed(inh_weight_19c),
            exc_weight => signed(exc_weight_19c),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19c,
            out_spike => out_spikes_inst(412)
        );

    neuron_19d : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19d,
            inh_weight => signed(inh_weight_19d),
            exc_weight => signed(exc_weight_19d),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19d,
            out_spike => out_spikes_inst(413)
        );

    neuron_19e : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19e,
            inh_weight => signed(inh_weight_19e),
            exc_weight => signed(exc_weight_19e),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19e,
            out_spike => out_spikes_inst(414)
        );

    neuron_19f : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_19f,
            inh_weight => signed(inh_weight_19f),
            exc_weight => signed(exc_weight_19f),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_19f,
            out_spike => out_spikes_inst(415)
        );

    neuron_1a0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a0,
            inh_weight => signed(inh_weight_1a0),
            exc_weight => signed(exc_weight_1a0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a0,
            out_spike => out_spikes_inst(416)
        );

    neuron_1a1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a1,
            inh_weight => signed(inh_weight_1a1),
            exc_weight => signed(exc_weight_1a1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a1,
            out_spike => out_spikes_inst(417)
        );

    neuron_1a2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a2,
            inh_weight => signed(inh_weight_1a2),
            exc_weight => signed(exc_weight_1a2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a2,
            out_spike => out_spikes_inst(418)
        );

    neuron_1a3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a3,
            inh_weight => signed(inh_weight_1a3),
            exc_weight => signed(exc_weight_1a3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a3,
            out_spike => out_spikes_inst(419)
        );

    neuron_1a4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a4,
            inh_weight => signed(inh_weight_1a4),
            exc_weight => signed(exc_weight_1a4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a4,
            out_spike => out_spikes_inst(420)
        );

    neuron_1a5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a5,
            inh_weight => signed(inh_weight_1a5),
            exc_weight => signed(exc_weight_1a5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a5,
            out_spike => out_spikes_inst(421)
        );

    neuron_1a6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a6,
            inh_weight => signed(inh_weight_1a6),
            exc_weight => signed(exc_weight_1a6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a6,
            out_spike => out_spikes_inst(422)
        );

    neuron_1a7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a7,
            inh_weight => signed(inh_weight_1a7),
            exc_weight => signed(exc_weight_1a7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a7,
            out_spike => out_spikes_inst(423)
        );

    neuron_1a8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a8,
            inh_weight => signed(inh_weight_1a8),
            exc_weight => signed(exc_weight_1a8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a8,
            out_spike => out_spikes_inst(424)
        );

    neuron_1a9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1a9,
            inh_weight => signed(inh_weight_1a9),
            exc_weight => signed(exc_weight_1a9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1a9,
            out_spike => out_spikes_inst(425)
        );

    neuron_1aa : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1aa,
            inh_weight => signed(inh_weight_1aa),
            exc_weight => signed(exc_weight_1aa),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1aa,
            out_spike => out_spikes_inst(426)
        );

    neuron_1ab : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ab,
            inh_weight => signed(inh_weight_1ab),
            exc_weight => signed(exc_weight_1ab),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ab,
            out_spike => out_spikes_inst(427)
        );

    neuron_1ac : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ac,
            inh_weight => signed(inh_weight_1ac),
            exc_weight => signed(exc_weight_1ac),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ac,
            out_spike => out_spikes_inst(428)
        );

    neuron_1ad : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ad,
            inh_weight => signed(inh_weight_1ad),
            exc_weight => signed(exc_weight_1ad),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ad,
            out_spike => out_spikes_inst(429)
        );

    neuron_1ae : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ae,
            inh_weight => signed(inh_weight_1ae),
            exc_weight => signed(exc_weight_1ae),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ae,
            out_spike => out_spikes_inst(430)
        );

    neuron_1af : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1af,
            inh_weight => signed(inh_weight_1af),
            exc_weight => signed(exc_weight_1af),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1af,
            out_spike => out_spikes_inst(431)
        );

    neuron_1b0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b0,
            inh_weight => signed(inh_weight_1b0),
            exc_weight => signed(exc_weight_1b0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b0,
            out_spike => out_spikes_inst(432)
        );

    neuron_1b1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b1,
            inh_weight => signed(inh_weight_1b1),
            exc_weight => signed(exc_weight_1b1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b1,
            out_spike => out_spikes_inst(433)
        );

    neuron_1b2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b2,
            inh_weight => signed(inh_weight_1b2),
            exc_weight => signed(exc_weight_1b2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b2,
            out_spike => out_spikes_inst(434)
        );

    neuron_1b3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b3,
            inh_weight => signed(inh_weight_1b3),
            exc_weight => signed(exc_weight_1b3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b3,
            out_spike => out_spikes_inst(435)
        );

    neuron_1b4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b4,
            inh_weight => signed(inh_weight_1b4),
            exc_weight => signed(exc_weight_1b4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b4,
            out_spike => out_spikes_inst(436)
        );

    neuron_1b5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b5,
            inh_weight => signed(inh_weight_1b5),
            exc_weight => signed(exc_weight_1b5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b5,
            out_spike => out_spikes_inst(437)
        );

    neuron_1b6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b6,
            inh_weight => signed(inh_weight_1b6),
            exc_weight => signed(exc_weight_1b6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b6,
            out_spike => out_spikes_inst(438)
        );

    neuron_1b7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b7,
            inh_weight => signed(inh_weight_1b7),
            exc_weight => signed(exc_weight_1b7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b7,
            out_spike => out_spikes_inst(439)
        );

    neuron_1b8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b8,
            inh_weight => signed(inh_weight_1b8),
            exc_weight => signed(exc_weight_1b8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b8,
            out_spike => out_spikes_inst(440)
        );

    neuron_1b9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1b9,
            inh_weight => signed(inh_weight_1b9),
            exc_weight => signed(exc_weight_1b9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1b9,
            out_spike => out_spikes_inst(441)
        );

    neuron_1ba : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ba,
            inh_weight => signed(inh_weight_1ba),
            exc_weight => signed(exc_weight_1ba),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ba,
            out_spike => out_spikes_inst(442)
        );

    neuron_1bb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1bb,
            inh_weight => signed(inh_weight_1bb),
            exc_weight => signed(exc_weight_1bb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1bb,
            out_spike => out_spikes_inst(443)
        );

    neuron_1bc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1bc,
            inh_weight => signed(inh_weight_1bc),
            exc_weight => signed(exc_weight_1bc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1bc,
            out_spike => out_spikes_inst(444)
        );

    neuron_1bd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1bd,
            inh_weight => signed(inh_weight_1bd),
            exc_weight => signed(exc_weight_1bd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1bd,
            out_spike => out_spikes_inst(445)
        );

    neuron_1be : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1be,
            inh_weight => signed(inh_weight_1be),
            exc_weight => signed(exc_weight_1be),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1be,
            out_spike => out_spikes_inst(446)
        );

    neuron_1bf : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1bf,
            inh_weight => signed(inh_weight_1bf),
            exc_weight => signed(exc_weight_1bf),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1bf,
            out_spike => out_spikes_inst(447)
        );

    neuron_1c0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c0,
            inh_weight => signed(inh_weight_1c0),
            exc_weight => signed(exc_weight_1c0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c0,
            out_spike => out_spikes_inst(448)
        );

    neuron_1c1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c1,
            inh_weight => signed(inh_weight_1c1),
            exc_weight => signed(exc_weight_1c1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c1,
            out_spike => out_spikes_inst(449)
        );

    neuron_1c2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c2,
            inh_weight => signed(inh_weight_1c2),
            exc_weight => signed(exc_weight_1c2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c2,
            out_spike => out_spikes_inst(450)
        );

    neuron_1c3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c3,
            inh_weight => signed(inh_weight_1c3),
            exc_weight => signed(exc_weight_1c3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c3,
            out_spike => out_spikes_inst(451)
        );

    neuron_1c4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c4,
            inh_weight => signed(inh_weight_1c4),
            exc_weight => signed(exc_weight_1c4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c4,
            out_spike => out_spikes_inst(452)
        );

    neuron_1c5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c5,
            inh_weight => signed(inh_weight_1c5),
            exc_weight => signed(exc_weight_1c5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c5,
            out_spike => out_spikes_inst(453)
        );

    neuron_1c6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c6,
            inh_weight => signed(inh_weight_1c6),
            exc_weight => signed(exc_weight_1c6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c6,
            out_spike => out_spikes_inst(454)
        );

    neuron_1c7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c7,
            inh_weight => signed(inh_weight_1c7),
            exc_weight => signed(exc_weight_1c7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c7,
            out_spike => out_spikes_inst(455)
        );

    neuron_1c8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c8,
            inh_weight => signed(inh_weight_1c8),
            exc_weight => signed(exc_weight_1c8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c8,
            out_spike => out_spikes_inst(456)
        );

    neuron_1c9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1c9,
            inh_weight => signed(inh_weight_1c9),
            exc_weight => signed(exc_weight_1c9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1c9,
            out_spike => out_spikes_inst(457)
        );

    neuron_1ca : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ca,
            inh_weight => signed(inh_weight_1ca),
            exc_weight => signed(exc_weight_1ca),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ca,
            out_spike => out_spikes_inst(458)
        );

    neuron_1cb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1cb,
            inh_weight => signed(inh_weight_1cb),
            exc_weight => signed(exc_weight_1cb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1cb,
            out_spike => out_spikes_inst(459)
        );

    neuron_1cc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1cc,
            inh_weight => signed(inh_weight_1cc),
            exc_weight => signed(exc_weight_1cc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1cc,
            out_spike => out_spikes_inst(460)
        );

    neuron_1cd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1cd,
            inh_weight => signed(inh_weight_1cd),
            exc_weight => signed(exc_weight_1cd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1cd,
            out_spike => out_spikes_inst(461)
        );

    neuron_1ce : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ce,
            inh_weight => signed(inh_weight_1ce),
            exc_weight => signed(exc_weight_1ce),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ce,
            out_spike => out_spikes_inst(462)
        );

    neuron_1cf : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1cf,
            inh_weight => signed(inh_weight_1cf),
            exc_weight => signed(exc_weight_1cf),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1cf,
            out_spike => out_spikes_inst(463)
        );

    neuron_1d0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d0,
            inh_weight => signed(inh_weight_1d0),
            exc_weight => signed(exc_weight_1d0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d0,
            out_spike => out_spikes_inst(464)
        );

    neuron_1d1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d1,
            inh_weight => signed(inh_weight_1d1),
            exc_weight => signed(exc_weight_1d1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d1,
            out_spike => out_spikes_inst(465)
        );

    neuron_1d2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d2,
            inh_weight => signed(inh_weight_1d2),
            exc_weight => signed(exc_weight_1d2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d2,
            out_spike => out_spikes_inst(466)
        );

    neuron_1d3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d3,
            inh_weight => signed(inh_weight_1d3),
            exc_weight => signed(exc_weight_1d3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d3,
            out_spike => out_spikes_inst(467)
        );

    neuron_1d4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d4,
            inh_weight => signed(inh_weight_1d4),
            exc_weight => signed(exc_weight_1d4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d4,
            out_spike => out_spikes_inst(468)
        );

    neuron_1d5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d5,
            inh_weight => signed(inh_weight_1d5),
            exc_weight => signed(exc_weight_1d5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d5,
            out_spike => out_spikes_inst(469)
        );

    neuron_1d6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d6,
            inh_weight => signed(inh_weight_1d6),
            exc_weight => signed(exc_weight_1d6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d6,
            out_spike => out_spikes_inst(470)
        );

    neuron_1d7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d7,
            inh_weight => signed(inh_weight_1d7),
            exc_weight => signed(exc_weight_1d7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d7,
            out_spike => out_spikes_inst(471)
        );

    neuron_1d8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d8,
            inh_weight => signed(inh_weight_1d8),
            exc_weight => signed(exc_weight_1d8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d8,
            out_spike => out_spikes_inst(472)
        );

    neuron_1d9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1d9,
            inh_weight => signed(inh_weight_1d9),
            exc_weight => signed(exc_weight_1d9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1d9,
            out_spike => out_spikes_inst(473)
        );

    neuron_1da : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1da,
            inh_weight => signed(inh_weight_1da),
            exc_weight => signed(exc_weight_1da),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1da,
            out_spike => out_spikes_inst(474)
        );

    neuron_1db : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1db,
            inh_weight => signed(inh_weight_1db),
            exc_weight => signed(exc_weight_1db),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1db,
            out_spike => out_spikes_inst(475)
        );

    neuron_1dc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1dc,
            inh_weight => signed(inh_weight_1dc),
            exc_weight => signed(exc_weight_1dc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1dc,
            out_spike => out_spikes_inst(476)
        );

    neuron_1dd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1dd,
            inh_weight => signed(inh_weight_1dd),
            exc_weight => signed(exc_weight_1dd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1dd,
            out_spike => out_spikes_inst(477)
        );

    neuron_1de : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1de,
            inh_weight => signed(inh_weight_1de),
            exc_weight => signed(exc_weight_1de),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1de,
            out_spike => out_spikes_inst(478)
        );

    neuron_1df : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1df,
            inh_weight => signed(inh_weight_1df),
            exc_weight => signed(exc_weight_1df),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1df,
            out_spike => out_spikes_inst(479)
        );

    neuron_1e0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e0,
            inh_weight => signed(inh_weight_1e0),
            exc_weight => signed(exc_weight_1e0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e0,
            out_spike => out_spikes_inst(480)
        );

    neuron_1e1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e1,
            inh_weight => signed(inh_weight_1e1),
            exc_weight => signed(exc_weight_1e1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e1,
            out_spike => out_spikes_inst(481)
        );

    neuron_1e2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e2,
            inh_weight => signed(inh_weight_1e2),
            exc_weight => signed(exc_weight_1e2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e2,
            out_spike => out_spikes_inst(482)
        );

    neuron_1e3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e3,
            inh_weight => signed(inh_weight_1e3),
            exc_weight => signed(exc_weight_1e3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e3,
            out_spike => out_spikes_inst(483)
        );

    neuron_1e4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e4,
            inh_weight => signed(inh_weight_1e4),
            exc_weight => signed(exc_weight_1e4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e4,
            out_spike => out_spikes_inst(484)
        );

    neuron_1e5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e5,
            inh_weight => signed(inh_weight_1e5),
            exc_weight => signed(exc_weight_1e5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e5,
            out_spike => out_spikes_inst(485)
        );

    neuron_1e6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e6,
            inh_weight => signed(inh_weight_1e6),
            exc_weight => signed(exc_weight_1e6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e6,
            out_spike => out_spikes_inst(486)
        );

    neuron_1e7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e7,
            inh_weight => signed(inh_weight_1e7),
            exc_weight => signed(exc_weight_1e7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e7,
            out_spike => out_spikes_inst(487)
        );

    neuron_1e8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e8,
            inh_weight => signed(inh_weight_1e8),
            exc_weight => signed(exc_weight_1e8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e8,
            out_spike => out_spikes_inst(488)
        );

    neuron_1e9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1e9,
            inh_weight => signed(inh_weight_1e9),
            exc_weight => signed(exc_weight_1e9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1e9,
            out_spike => out_spikes_inst(489)
        );

    neuron_1ea : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ea,
            inh_weight => signed(inh_weight_1ea),
            exc_weight => signed(exc_weight_1ea),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ea,
            out_spike => out_spikes_inst(490)
        );

    neuron_1eb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1eb,
            inh_weight => signed(inh_weight_1eb),
            exc_weight => signed(exc_weight_1eb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1eb,
            out_spike => out_spikes_inst(491)
        );

    neuron_1ec : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ec,
            inh_weight => signed(inh_weight_1ec),
            exc_weight => signed(exc_weight_1ec),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ec,
            out_spike => out_spikes_inst(492)
        );

    neuron_1ed : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ed,
            inh_weight => signed(inh_weight_1ed),
            exc_weight => signed(exc_weight_1ed),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ed,
            out_spike => out_spikes_inst(493)
        );

    neuron_1ee : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ee,
            inh_weight => signed(inh_weight_1ee),
            exc_weight => signed(exc_weight_1ee),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ee,
            out_spike => out_spikes_inst(494)
        );

    neuron_1ef : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ef,
            inh_weight => signed(inh_weight_1ef),
            exc_weight => signed(exc_weight_1ef),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ef,
            out_spike => out_spikes_inst(495)
        );

    neuron_1f0 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f0,
            inh_weight => signed(inh_weight_1f0),
            exc_weight => signed(exc_weight_1f0),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f0,
            out_spike => out_spikes_inst(496)
        );

    neuron_1f1 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f1,
            inh_weight => signed(inh_weight_1f1),
            exc_weight => signed(exc_weight_1f1),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f1,
            out_spike => out_spikes_inst(497)
        );

    neuron_1f2 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f2,
            inh_weight => signed(inh_weight_1f2),
            exc_weight => signed(exc_weight_1f2),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f2,
            out_spike => out_spikes_inst(498)
        );

    neuron_1f3 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f3,
            inh_weight => signed(inh_weight_1f3),
            exc_weight => signed(exc_weight_1f3),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f3,
            out_spike => out_spikes_inst(499)
        );

    neuron_1f4 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f4,
            inh_weight => signed(inh_weight_1f4),
            exc_weight => signed(exc_weight_1f4),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f4,
            out_spike => out_spikes_inst(500)
        );

    neuron_1f5 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f5,
            inh_weight => signed(inh_weight_1f5),
            exc_weight => signed(exc_weight_1f5),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f5,
            out_spike => out_spikes_inst(501)
        );

    neuron_1f6 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f6,
            inh_weight => signed(inh_weight_1f6),
            exc_weight => signed(exc_weight_1f6),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f6,
            out_spike => out_spikes_inst(502)
        );

    neuron_1f7 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f7,
            inh_weight => signed(inh_weight_1f7),
            exc_weight => signed(exc_weight_1f7),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f7,
            out_spike => out_spikes_inst(503)
        );

    neuron_1f8 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f8,
            inh_weight => signed(inh_weight_1f8),
            exc_weight => signed(exc_weight_1f8),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f8,
            out_spike => out_spikes_inst(504)
        );

    neuron_1f9 : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1f9,
            inh_weight => signed(inh_weight_1f9),
            exc_weight => signed(exc_weight_1f9),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1f9,
            out_spike => out_spikes_inst(505)
        );

    neuron_1fa : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1fa,
            inh_weight => signed(inh_weight_1fa),
            exc_weight => signed(exc_weight_1fa),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1fa,
            out_spike => out_spikes_inst(506)
        );

    neuron_1fb : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1fb,
            inh_weight => signed(inh_weight_1fb),
            exc_weight => signed(exc_weight_1fb),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1fb,
            out_spike => out_spikes_inst(507)
        );

    neuron_1fc : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1fc,
            inh_weight => signed(inh_weight_1fc),
            exc_weight => signed(exc_weight_1fc),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1fc,
            out_spike => out_spikes_inst(508)
        );

    neuron_1fd : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1fd,
            inh_weight => signed(inh_weight_1fd),
            exc_weight => signed(exc_weight_1fd),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1fd,
            out_spike => out_spikes_inst(509)
        );

    neuron_1fe : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1fe,
            inh_weight => signed(inh_weight_1fe),
            exc_weight => signed(exc_weight_1fe),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1fe,
            out_spike => out_spikes_inst(510)
        );

    neuron_1ff : neuron_subtractive
        generic map(
            neuron_bit_width => neuron_bit_width,
            shift => shift
        )
        port map(
            v_th => v_th_1ff,
            inh_weight => signed(inh_weight_1ff),
            exc_weight => signed(exc_weight_1ff),
            clk => clk,
            rst_n => rst_n,
            restart => neuron_restart,
            exc => exc,
            inh => inh,
            exc_spike => exc_spike,
            inh_spike => inh_spike,
            neuron_ready => neuron_ready_1ff,
            out_spike => out_spikes_inst(511)
        );

    exc_mem : rom_1024x512_exclif2
        port map(
            clka => clk,
            addra => exc_addr,
            dout_00 => exc_weight_00,
            dout_01 => exc_weight_01,
            dout_02 => exc_weight_02,
            dout_03 => exc_weight_03,
            dout_04 => exc_weight_04,
            dout_05 => exc_weight_05,
            dout_06 => exc_weight_06,
            dout_07 => exc_weight_07,
            dout_08 => exc_weight_08,
            dout_09 => exc_weight_09,
            dout_0a => exc_weight_0a,
            dout_0b => exc_weight_0b,
            dout_0c => exc_weight_0c,
            dout_0d => exc_weight_0d,
            dout_0e => exc_weight_0e,
            dout_0f => exc_weight_0f,
            dout_10 => exc_weight_10,
            dout_11 => exc_weight_11,
            dout_12 => exc_weight_12,
            dout_13 => exc_weight_13,
            dout_14 => exc_weight_14,
            dout_15 => exc_weight_15,
            dout_16 => exc_weight_16,
            dout_17 => exc_weight_17,
            dout_18 => exc_weight_18,
            dout_19 => exc_weight_19,
            dout_1a => exc_weight_1a,
            dout_1b => exc_weight_1b,
            dout_1c => exc_weight_1c,
            dout_1d => exc_weight_1d,
            dout_1e => exc_weight_1e,
            dout_1f => exc_weight_1f,
            dout_20 => exc_weight_20,
            dout_21 => exc_weight_21,
            dout_22 => exc_weight_22,
            dout_23 => exc_weight_23,
            dout_24 => exc_weight_24,
            dout_25 => exc_weight_25,
            dout_26 => exc_weight_26,
            dout_27 => exc_weight_27,
            dout_28 => exc_weight_28,
            dout_29 => exc_weight_29,
            dout_2a => exc_weight_2a,
            dout_2b => exc_weight_2b,
            dout_2c => exc_weight_2c,
            dout_2d => exc_weight_2d,
            dout_2e => exc_weight_2e,
            dout_2f => exc_weight_2f,
            dout_30 => exc_weight_30,
            dout_31 => exc_weight_31,
            dout_32 => exc_weight_32,
            dout_33 => exc_weight_33,
            dout_34 => exc_weight_34,
            dout_35 => exc_weight_35,
            dout_36 => exc_weight_36,
            dout_37 => exc_weight_37,
            dout_38 => exc_weight_38,
            dout_39 => exc_weight_39,
            dout_3a => exc_weight_3a,
            dout_3b => exc_weight_3b,
            dout_3c => exc_weight_3c,
            dout_3d => exc_weight_3d,
            dout_3e => exc_weight_3e,
            dout_3f => exc_weight_3f,
            dout_40 => exc_weight_40,
            dout_41 => exc_weight_41,
            dout_42 => exc_weight_42,
            dout_43 => exc_weight_43,
            dout_44 => exc_weight_44,
            dout_45 => exc_weight_45,
            dout_46 => exc_weight_46,
            dout_47 => exc_weight_47,
            dout_48 => exc_weight_48,
            dout_49 => exc_weight_49,
            dout_4a => exc_weight_4a,
            dout_4b => exc_weight_4b,
            dout_4c => exc_weight_4c,
            dout_4d => exc_weight_4d,
            dout_4e => exc_weight_4e,
            dout_4f => exc_weight_4f,
            dout_50 => exc_weight_50,
            dout_51 => exc_weight_51,
            dout_52 => exc_weight_52,
            dout_53 => exc_weight_53,
            dout_54 => exc_weight_54,
            dout_55 => exc_weight_55,
            dout_56 => exc_weight_56,
            dout_57 => exc_weight_57,
            dout_58 => exc_weight_58,
            dout_59 => exc_weight_59,
            dout_5a => exc_weight_5a,
            dout_5b => exc_weight_5b,
            dout_5c => exc_weight_5c,
            dout_5d => exc_weight_5d,
            dout_5e => exc_weight_5e,
            dout_5f => exc_weight_5f,
            dout_60 => exc_weight_60,
            dout_61 => exc_weight_61,
            dout_62 => exc_weight_62,
            dout_63 => exc_weight_63,
            dout_64 => exc_weight_64,
            dout_65 => exc_weight_65,
            dout_66 => exc_weight_66,
            dout_67 => exc_weight_67,
            dout_68 => exc_weight_68,
            dout_69 => exc_weight_69,
            dout_6a => exc_weight_6a,
            dout_6b => exc_weight_6b,
            dout_6c => exc_weight_6c,
            dout_6d => exc_weight_6d,
            dout_6e => exc_weight_6e,
            dout_6f => exc_weight_6f,
            dout_70 => exc_weight_70,
            dout_71 => exc_weight_71,
            dout_72 => exc_weight_72,
            dout_73 => exc_weight_73,
            dout_74 => exc_weight_74,
            dout_75 => exc_weight_75,
            dout_76 => exc_weight_76,
            dout_77 => exc_weight_77,
            dout_78 => exc_weight_78,
            dout_79 => exc_weight_79,
            dout_7a => exc_weight_7a,
            dout_7b => exc_weight_7b,
            dout_7c => exc_weight_7c,
            dout_7d => exc_weight_7d,
            dout_7e => exc_weight_7e,
            dout_7f => exc_weight_7f,
            dout_80 => exc_weight_80,
            dout_81 => exc_weight_81,
            dout_82 => exc_weight_82,
            dout_83 => exc_weight_83,
            dout_84 => exc_weight_84,
            dout_85 => exc_weight_85,
            dout_86 => exc_weight_86,
            dout_87 => exc_weight_87,
            dout_88 => exc_weight_88,
            dout_89 => exc_weight_89,
            dout_8a => exc_weight_8a,
            dout_8b => exc_weight_8b,
            dout_8c => exc_weight_8c,
            dout_8d => exc_weight_8d,
            dout_8e => exc_weight_8e,
            dout_8f => exc_weight_8f,
            dout_90 => exc_weight_90,
            dout_91 => exc_weight_91,
            dout_92 => exc_weight_92,
            dout_93 => exc_weight_93,
            dout_94 => exc_weight_94,
            dout_95 => exc_weight_95,
            dout_96 => exc_weight_96,
            dout_97 => exc_weight_97,
            dout_98 => exc_weight_98,
            dout_99 => exc_weight_99,
            dout_9a => exc_weight_9a,
            dout_9b => exc_weight_9b,
            dout_9c => exc_weight_9c,
            dout_9d => exc_weight_9d,
            dout_9e => exc_weight_9e,
            dout_9f => exc_weight_9f,
            dout_a0 => exc_weight_a0,
            dout_a1 => exc_weight_a1,
            dout_a2 => exc_weight_a2,
            dout_a3 => exc_weight_a3,
            dout_a4 => exc_weight_a4,
            dout_a5 => exc_weight_a5,
            dout_a6 => exc_weight_a6,
            dout_a7 => exc_weight_a7,
            dout_a8 => exc_weight_a8,
            dout_a9 => exc_weight_a9,
            dout_aa => exc_weight_aa,
            dout_ab => exc_weight_ab,
            dout_ac => exc_weight_ac,
            dout_ad => exc_weight_ad,
            dout_ae => exc_weight_ae,
            dout_af => exc_weight_af,
            dout_b0 => exc_weight_b0,
            dout_b1 => exc_weight_b1,
            dout_b2 => exc_weight_b2,
            dout_b3 => exc_weight_b3,
            dout_b4 => exc_weight_b4,
            dout_b5 => exc_weight_b5,
            dout_b6 => exc_weight_b6,
            dout_b7 => exc_weight_b7,
            dout_b8 => exc_weight_b8,
            dout_b9 => exc_weight_b9,
            dout_ba => exc_weight_ba,
            dout_bb => exc_weight_bb,
            dout_bc => exc_weight_bc,
            dout_bd => exc_weight_bd,
            dout_be => exc_weight_be,
            dout_bf => exc_weight_bf,
            dout_c0 => exc_weight_c0,
            dout_c1 => exc_weight_c1,
            dout_c2 => exc_weight_c2,
            dout_c3 => exc_weight_c3,
            dout_c4 => exc_weight_c4,
            dout_c5 => exc_weight_c5,
            dout_c6 => exc_weight_c6,
            dout_c7 => exc_weight_c7,
            dout_c8 => exc_weight_c8,
            dout_c9 => exc_weight_c9,
            dout_ca => exc_weight_ca,
            dout_cb => exc_weight_cb,
            dout_cc => exc_weight_cc,
            dout_cd => exc_weight_cd,
            dout_ce => exc_weight_ce,
            dout_cf => exc_weight_cf,
            dout_d0 => exc_weight_d0,
            dout_d1 => exc_weight_d1,
            dout_d2 => exc_weight_d2,
            dout_d3 => exc_weight_d3,
            dout_d4 => exc_weight_d4,
            dout_d5 => exc_weight_d5,
            dout_d6 => exc_weight_d6,
            dout_d7 => exc_weight_d7,
            dout_d8 => exc_weight_d8,
            dout_d9 => exc_weight_d9,
            dout_da => exc_weight_da,
            dout_db => exc_weight_db,
            dout_dc => exc_weight_dc,
            dout_dd => exc_weight_dd,
            dout_de => exc_weight_de,
            dout_df => exc_weight_df,
            dout_e0 => exc_weight_e0,
            dout_e1 => exc_weight_e1,
            dout_e2 => exc_weight_e2,
            dout_e3 => exc_weight_e3,
            dout_e4 => exc_weight_e4,
            dout_e5 => exc_weight_e5,
            dout_e6 => exc_weight_e6,
            dout_e7 => exc_weight_e7,
            dout_e8 => exc_weight_e8,
            dout_e9 => exc_weight_e9,
            dout_ea => exc_weight_ea,
            dout_eb => exc_weight_eb,
            dout_ec => exc_weight_ec,
            dout_ed => exc_weight_ed,
            dout_ee => exc_weight_ee,
            dout_ef => exc_weight_ef,
            dout_f0 => exc_weight_f0,
            dout_f1 => exc_weight_f1,
            dout_f2 => exc_weight_f2,
            dout_f3 => exc_weight_f3,
            dout_f4 => exc_weight_f4,
            dout_f5 => exc_weight_f5,
            dout_f6 => exc_weight_f6,
            dout_f7 => exc_weight_f7,
            dout_f8 => exc_weight_f8,
            dout_f9 => exc_weight_f9,
            dout_fa => exc_weight_fa,
            dout_fb => exc_weight_fb,
            dout_fc => exc_weight_fc,
            dout_fd => exc_weight_fd,
            dout_fe => exc_weight_fe,
            dout_ff => exc_weight_ff,
            dout_100 => exc_weight_100,
            dout_101 => exc_weight_101,
            dout_102 => exc_weight_102,
            dout_103 => exc_weight_103,
            dout_104 => exc_weight_104,
            dout_105 => exc_weight_105,
            dout_106 => exc_weight_106,
            dout_107 => exc_weight_107,
            dout_108 => exc_weight_108,
            dout_109 => exc_weight_109,
            dout_10a => exc_weight_10a,
            dout_10b => exc_weight_10b,
            dout_10c => exc_weight_10c,
            dout_10d => exc_weight_10d,
            dout_10e => exc_weight_10e,
            dout_10f => exc_weight_10f,
            dout_110 => exc_weight_110,
            dout_111 => exc_weight_111,
            dout_112 => exc_weight_112,
            dout_113 => exc_weight_113,
            dout_114 => exc_weight_114,
            dout_115 => exc_weight_115,
            dout_116 => exc_weight_116,
            dout_117 => exc_weight_117,
            dout_118 => exc_weight_118,
            dout_119 => exc_weight_119,
            dout_11a => exc_weight_11a,
            dout_11b => exc_weight_11b,
            dout_11c => exc_weight_11c,
            dout_11d => exc_weight_11d,
            dout_11e => exc_weight_11e,
            dout_11f => exc_weight_11f,
            dout_120 => exc_weight_120,
            dout_121 => exc_weight_121,
            dout_122 => exc_weight_122,
            dout_123 => exc_weight_123,
            dout_124 => exc_weight_124,
            dout_125 => exc_weight_125,
            dout_126 => exc_weight_126,
            dout_127 => exc_weight_127,
            dout_128 => exc_weight_128,
            dout_129 => exc_weight_129,
            dout_12a => exc_weight_12a,
            dout_12b => exc_weight_12b,
            dout_12c => exc_weight_12c,
            dout_12d => exc_weight_12d,
            dout_12e => exc_weight_12e,
            dout_12f => exc_weight_12f,
            dout_130 => exc_weight_130,
            dout_131 => exc_weight_131,
            dout_132 => exc_weight_132,
            dout_133 => exc_weight_133,
            dout_134 => exc_weight_134,
            dout_135 => exc_weight_135,
            dout_136 => exc_weight_136,
            dout_137 => exc_weight_137,
            dout_138 => exc_weight_138,
            dout_139 => exc_weight_139,
            dout_13a => exc_weight_13a,
            dout_13b => exc_weight_13b,
            dout_13c => exc_weight_13c,
            dout_13d => exc_weight_13d,
            dout_13e => exc_weight_13e,
            dout_13f => exc_weight_13f,
            dout_140 => exc_weight_140,
            dout_141 => exc_weight_141,
            dout_142 => exc_weight_142,
            dout_143 => exc_weight_143,
            dout_144 => exc_weight_144,
            dout_145 => exc_weight_145,
            dout_146 => exc_weight_146,
            dout_147 => exc_weight_147,
            dout_148 => exc_weight_148,
            dout_149 => exc_weight_149,
            dout_14a => exc_weight_14a,
            dout_14b => exc_weight_14b,
            dout_14c => exc_weight_14c,
            dout_14d => exc_weight_14d,
            dout_14e => exc_weight_14e,
            dout_14f => exc_weight_14f,
            dout_150 => exc_weight_150,
            dout_151 => exc_weight_151,
            dout_152 => exc_weight_152,
            dout_153 => exc_weight_153,
            dout_154 => exc_weight_154,
            dout_155 => exc_weight_155,
            dout_156 => exc_weight_156,
            dout_157 => exc_weight_157,
            dout_158 => exc_weight_158,
            dout_159 => exc_weight_159,
            dout_15a => exc_weight_15a,
            dout_15b => exc_weight_15b,
            dout_15c => exc_weight_15c,
            dout_15d => exc_weight_15d,
            dout_15e => exc_weight_15e,
            dout_15f => exc_weight_15f,
            dout_160 => exc_weight_160,
            dout_161 => exc_weight_161,
            dout_162 => exc_weight_162,
            dout_163 => exc_weight_163,
            dout_164 => exc_weight_164,
            dout_165 => exc_weight_165,
            dout_166 => exc_weight_166,
            dout_167 => exc_weight_167,
            dout_168 => exc_weight_168,
            dout_169 => exc_weight_169,
            dout_16a => exc_weight_16a,
            dout_16b => exc_weight_16b,
            dout_16c => exc_weight_16c,
            dout_16d => exc_weight_16d,
            dout_16e => exc_weight_16e,
            dout_16f => exc_weight_16f,
            dout_170 => exc_weight_170,
            dout_171 => exc_weight_171,
            dout_172 => exc_weight_172,
            dout_173 => exc_weight_173,
            dout_174 => exc_weight_174,
            dout_175 => exc_weight_175,
            dout_176 => exc_weight_176,
            dout_177 => exc_weight_177,
            dout_178 => exc_weight_178,
            dout_179 => exc_weight_179,
            dout_17a => exc_weight_17a,
            dout_17b => exc_weight_17b,
            dout_17c => exc_weight_17c,
            dout_17d => exc_weight_17d,
            dout_17e => exc_weight_17e,
            dout_17f => exc_weight_17f,
            dout_180 => exc_weight_180,
            dout_181 => exc_weight_181,
            dout_182 => exc_weight_182,
            dout_183 => exc_weight_183,
            dout_184 => exc_weight_184,
            dout_185 => exc_weight_185,
            dout_186 => exc_weight_186,
            dout_187 => exc_weight_187,
            dout_188 => exc_weight_188,
            dout_189 => exc_weight_189,
            dout_18a => exc_weight_18a,
            dout_18b => exc_weight_18b,
            dout_18c => exc_weight_18c,
            dout_18d => exc_weight_18d,
            dout_18e => exc_weight_18e,
            dout_18f => exc_weight_18f,
            dout_190 => exc_weight_190,
            dout_191 => exc_weight_191,
            dout_192 => exc_weight_192,
            dout_193 => exc_weight_193,
            dout_194 => exc_weight_194,
            dout_195 => exc_weight_195,
            dout_196 => exc_weight_196,
            dout_197 => exc_weight_197,
            dout_198 => exc_weight_198,
            dout_199 => exc_weight_199,
            dout_19a => exc_weight_19a,
            dout_19b => exc_weight_19b,
            dout_19c => exc_weight_19c,
            dout_19d => exc_weight_19d,
            dout_19e => exc_weight_19e,
            dout_19f => exc_weight_19f,
            dout_1a0 => exc_weight_1a0,
            dout_1a1 => exc_weight_1a1,
            dout_1a2 => exc_weight_1a2,
            dout_1a3 => exc_weight_1a3,
            dout_1a4 => exc_weight_1a4,
            dout_1a5 => exc_weight_1a5,
            dout_1a6 => exc_weight_1a6,
            dout_1a7 => exc_weight_1a7,
            dout_1a8 => exc_weight_1a8,
            dout_1a9 => exc_weight_1a9,
            dout_1aa => exc_weight_1aa,
            dout_1ab => exc_weight_1ab,
            dout_1ac => exc_weight_1ac,
            dout_1ad => exc_weight_1ad,
            dout_1ae => exc_weight_1ae,
            dout_1af => exc_weight_1af,
            dout_1b0 => exc_weight_1b0,
            dout_1b1 => exc_weight_1b1,
            dout_1b2 => exc_weight_1b2,
            dout_1b3 => exc_weight_1b3,
            dout_1b4 => exc_weight_1b4,
            dout_1b5 => exc_weight_1b5,
            dout_1b6 => exc_weight_1b6,
            dout_1b7 => exc_weight_1b7,
            dout_1b8 => exc_weight_1b8,
            dout_1b9 => exc_weight_1b9,
            dout_1ba => exc_weight_1ba,
            dout_1bb => exc_weight_1bb,
            dout_1bc => exc_weight_1bc,
            dout_1bd => exc_weight_1bd,
            dout_1be => exc_weight_1be,
            dout_1bf => exc_weight_1bf,
            dout_1c0 => exc_weight_1c0,
            dout_1c1 => exc_weight_1c1,
            dout_1c2 => exc_weight_1c2,
            dout_1c3 => exc_weight_1c3,
            dout_1c4 => exc_weight_1c4,
            dout_1c5 => exc_weight_1c5,
            dout_1c6 => exc_weight_1c6,
            dout_1c7 => exc_weight_1c7,
            dout_1c8 => exc_weight_1c8,
            dout_1c9 => exc_weight_1c9,
            dout_1ca => exc_weight_1ca,
            dout_1cb => exc_weight_1cb,
            dout_1cc => exc_weight_1cc,
            dout_1cd => exc_weight_1cd,
            dout_1ce => exc_weight_1ce,
            dout_1cf => exc_weight_1cf,
            dout_1d0 => exc_weight_1d0,
            dout_1d1 => exc_weight_1d1,
            dout_1d2 => exc_weight_1d2,
            dout_1d3 => exc_weight_1d3,
            dout_1d4 => exc_weight_1d4,
            dout_1d5 => exc_weight_1d5,
            dout_1d6 => exc_weight_1d6,
            dout_1d7 => exc_weight_1d7,
            dout_1d8 => exc_weight_1d8,
            dout_1d9 => exc_weight_1d9,
            dout_1da => exc_weight_1da,
            dout_1db => exc_weight_1db,
            dout_1dc => exc_weight_1dc,
            dout_1dd => exc_weight_1dd,
            dout_1de => exc_weight_1de,
            dout_1df => exc_weight_1df,
            dout_1e0 => exc_weight_1e0,
            dout_1e1 => exc_weight_1e1,
            dout_1e2 => exc_weight_1e2,
            dout_1e3 => exc_weight_1e3,
            dout_1e4 => exc_weight_1e4,
            dout_1e5 => exc_weight_1e5,
            dout_1e6 => exc_weight_1e6,
            dout_1e7 => exc_weight_1e7,
            dout_1e8 => exc_weight_1e8,
            dout_1e9 => exc_weight_1e9,
            dout_1ea => exc_weight_1ea,
            dout_1eb => exc_weight_1eb,
            dout_1ec => exc_weight_1ec,
            dout_1ed => exc_weight_1ed,
            dout_1ee => exc_weight_1ee,
            dout_1ef => exc_weight_1ef,
            dout_1f0 => exc_weight_1f0,
            dout_1f1 => exc_weight_1f1,
            dout_1f2 => exc_weight_1f2,
            dout_1f3 => exc_weight_1f3,
            dout_1f4 => exc_weight_1f4,
            dout_1f5 => exc_weight_1f5,
            dout_1f6 => exc_weight_1f6,
            dout_1f7 => exc_weight_1f7,
            dout_1f8 => exc_weight_1f8,
            dout_1f9 => exc_weight_1f9,
            dout_1fa => exc_weight_1fa,
            dout_1fb => exc_weight_1fb,
            dout_1fc => exc_weight_1fc,
            dout_1fd => exc_weight_1fd,
            dout_1fe => exc_weight_1fe,
            dout_1ff => exc_weight_1ff
        );

    exc_addr_conv : addr_converter
        generic map(
            N => exc_cnt_bitwidth
        )
        port map(
            addr_in => exc_cnt,
            addr_out => exc_addr
        );

    inh_mem : rom_512x512_inhlif2
        port map(
            clka => clk,
            addra => inh_addr,
            dout_00 => inh_weight_00,
            dout_01 => inh_weight_01,
            dout_02 => inh_weight_02,
            dout_03 => inh_weight_03,
            dout_04 => inh_weight_04,
            dout_05 => inh_weight_05,
            dout_06 => inh_weight_06,
            dout_07 => inh_weight_07,
            dout_08 => inh_weight_08,
            dout_09 => inh_weight_09,
            dout_0a => inh_weight_0a,
            dout_0b => inh_weight_0b,
            dout_0c => inh_weight_0c,
            dout_0d => inh_weight_0d,
            dout_0e => inh_weight_0e,
            dout_0f => inh_weight_0f,
            dout_10 => inh_weight_10,
            dout_11 => inh_weight_11,
            dout_12 => inh_weight_12,
            dout_13 => inh_weight_13,
            dout_14 => inh_weight_14,
            dout_15 => inh_weight_15,
            dout_16 => inh_weight_16,
            dout_17 => inh_weight_17,
            dout_18 => inh_weight_18,
            dout_19 => inh_weight_19,
            dout_1a => inh_weight_1a,
            dout_1b => inh_weight_1b,
            dout_1c => inh_weight_1c,
            dout_1d => inh_weight_1d,
            dout_1e => inh_weight_1e,
            dout_1f => inh_weight_1f,
            dout_20 => inh_weight_20,
            dout_21 => inh_weight_21,
            dout_22 => inh_weight_22,
            dout_23 => inh_weight_23,
            dout_24 => inh_weight_24,
            dout_25 => inh_weight_25,
            dout_26 => inh_weight_26,
            dout_27 => inh_weight_27,
            dout_28 => inh_weight_28,
            dout_29 => inh_weight_29,
            dout_2a => inh_weight_2a,
            dout_2b => inh_weight_2b,
            dout_2c => inh_weight_2c,
            dout_2d => inh_weight_2d,
            dout_2e => inh_weight_2e,
            dout_2f => inh_weight_2f,
            dout_30 => inh_weight_30,
            dout_31 => inh_weight_31,
            dout_32 => inh_weight_32,
            dout_33 => inh_weight_33,
            dout_34 => inh_weight_34,
            dout_35 => inh_weight_35,
            dout_36 => inh_weight_36,
            dout_37 => inh_weight_37,
            dout_38 => inh_weight_38,
            dout_39 => inh_weight_39,
            dout_3a => inh_weight_3a,
            dout_3b => inh_weight_3b,
            dout_3c => inh_weight_3c,
            dout_3d => inh_weight_3d,
            dout_3e => inh_weight_3e,
            dout_3f => inh_weight_3f,
            dout_40 => inh_weight_40,
            dout_41 => inh_weight_41,
            dout_42 => inh_weight_42,
            dout_43 => inh_weight_43,
            dout_44 => inh_weight_44,
            dout_45 => inh_weight_45,
            dout_46 => inh_weight_46,
            dout_47 => inh_weight_47,
            dout_48 => inh_weight_48,
            dout_49 => inh_weight_49,
            dout_4a => inh_weight_4a,
            dout_4b => inh_weight_4b,
            dout_4c => inh_weight_4c,
            dout_4d => inh_weight_4d,
            dout_4e => inh_weight_4e,
            dout_4f => inh_weight_4f,
            dout_50 => inh_weight_50,
            dout_51 => inh_weight_51,
            dout_52 => inh_weight_52,
            dout_53 => inh_weight_53,
            dout_54 => inh_weight_54,
            dout_55 => inh_weight_55,
            dout_56 => inh_weight_56,
            dout_57 => inh_weight_57,
            dout_58 => inh_weight_58,
            dout_59 => inh_weight_59,
            dout_5a => inh_weight_5a,
            dout_5b => inh_weight_5b,
            dout_5c => inh_weight_5c,
            dout_5d => inh_weight_5d,
            dout_5e => inh_weight_5e,
            dout_5f => inh_weight_5f,
            dout_60 => inh_weight_60,
            dout_61 => inh_weight_61,
            dout_62 => inh_weight_62,
            dout_63 => inh_weight_63,
            dout_64 => inh_weight_64,
            dout_65 => inh_weight_65,
            dout_66 => inh_weight_66,
            dout_67 => inh_weight_67,
            dout_68 => inh_weight_68,
            dout_69 => inh_weight_69,
            dout_6a => inh_weight_6a,
            dout_6b => inh_weight_6b,
            dout_6c => inh_weight_6c,
            dout_6d => inh_weight_6d,
            dout_6e => inh_weight_6e,
            dout_6f => inh_weight_6f,
            dout_70 => inh_weight_70,
            dout_71 => inh_weight_71,
            dout_72 => inh_weight_72,
            dout_73 => inh_weight_73,
            dout_74 => inh_weight_74,
            dout_75 => inh_weight_75,
            dout_76 => inh_weight_76,
            dout_77 => inh_weight_77,
            dout_78 => inh_weight_78,
            dout_79 => inh_weight_79,
            dout_7a => inh_weight_7a,
            dout_7b => inh_weight_7b,
            dout_7c => inh_weight_7c,
            dout_7d => inh_weight_7d,
            dout_7e => inh_weight_7e,
            dout_7f => inh_weight_7f,
            dout_80 => inh_weight_80,
            dout_81 => inh_weight_81,
            dout_82 => inh_weight_82,
            dout_83 => inh_weight_83,
            dout_84 => inh_weight_84,
            dout_85 => inh_weight_85,
            dout_86 => inh_weight_86,
            dout_87 => inh_weight_87,
            dout_88 => inh_weight_88,
            dout_89 => inh_weight_89,
            dout_8a => inh_weight_8a,
            dout_8b => inh_weight_8b,
            dout_8c => inh_weight_8c,
            dout_8d => inh_weight_8d,
            dout_8e => inh_weight_8e,
            dout_8f => inh_weight_8f,
            dout_90 => inh_weight_90,
            dout_91 => inh_weight_91,
            dout_92 => inh_weight_92,
            dout_93 => inh_weight_93,
            dout_94 => inh_weight_94,
            dout_95 => inh_weight_95,
            dout_96 => inh_weight_96,
            dout_97 => inh_weight_97,
            dout_98 => inh_weight_98,
            dout_99 => inh_weight_99,
            dout_9a => inh_weight_9a,
            dout_9b => inh_weight_9b,
            dout_9c => inh_weight_9c,
            dout_9d => inh_weight_9d,
            dout_9e => inh_weight_9e,
            dout_9f => inh_weight_9f,
            dout_a0 => inh_weight_a0,
            dout_a1 => inh_weight_a1,
            dout_a2 => inh_weight_a2,
            dout_a3 => inh_weight_a3,
            dout_a4 => inh_weight_a4,
            dout_a5 => inh_weight_a5,
            dout_a6 => inh_weight_a6,
            dout_a7 => inh_weight_a7,
            dout_a8 => inh_weight_a8,
            dout_a9 => inh_weight_a9,
            dout_aa => inh_weight_aa,
            dout_ab => inh_weight_ab,
            dout_ac => inh_weight_ac,
            dout_ad => inh_weight_ad,
            dout_ae => inh_weight_ae,
            dout_af => inh_weight_af,
            dout_b0 => inh_weight_b0,
            dout_b1 => inh_weight_b1,
            dout_b2 => inh_weight_b2,
            dout_b3 => inh_weight_b3,
            dout_b4 => inh_weight_b4,
            dout_b5 => inh_weight_b5,
            dout_b6 => inh_weight_b6,
            dout_b7 => inh_weight_b7,
            dout_b8 => inh_weight_b8,
            dout_b9 => inh_weight_b9,
            dout_ba => inh_weight_ba,
            dout_bb => inh_weight_bb,
            dout_bc => inh_weight_bc,
            dout_bd => inh_weight_bd,
            dout_be => inh_weight_be,
            dout_bf => inh_weight_bf,
            dout_c0 => inh_weight_c0,
            dout_c1 => inh_weight_c1,
            dout_c2 => inh_weight_c2,
            dout_c3 => inh_weight_c3,
            dout_c4 => inh_weight_c4,
            dout_c5 => inh_weight_c5,
            dout_c6 => inh_weight_c6,
            dout_c7 => inh_weight_c7,
            dout_c8 => inh_weight_c8,
            dout_c9 => inh_weight_c9,
            dout_ca => inh_weight_ca,
            dout_cb => inh_weight_cb,
            dout_cc => inh_weight_cc,
            dout_cd => inh_weight_cd,
            dout_ce => inh_weight_ce,
            dout_cf => inh_weight_cf,
            dout_d0 => inh_weight_d0,
            dout_d1 => inh_weight_d1,
            dout_d2 => inh_weight_d2,
            dout_d3 => inh_weight_d3,
            dout_d4 => inh_weight_d4,
            dout_d5 => inh_weight_d5,
            dout_d6 => inh_weight_d6,
            dout_d7 => inh_weight_d7,
            dout_d8 => inh_weight_d8,
            dout_d9 => inh_weight_d9,
            dout_da => inh_weight_da,
            dout_db => inh_weight_db,
            dout_dc => inh_weight_dc,
            dout_dd => inh_weight_dd,
            dout_de => inh_weight_de,
            dout_df => inh_weight_df,
            dout_e0 => inh_weight_e0,
            dout_e1 => inh_weight_e1,
            dout_e2 => inh_weight_e2,
            dout_e3 => inh_weight_e3,
            dout_e4 => inh_weight_e4,
            dout_e5 => inh_weight_e5,
            dout_e6 => inh_weight_e6,
            dout_e7 => inh_weight_e7,
            dout_e8 => inh_weight_e8,
            dout_e9 => inh_weight_e9,
            dout_ea => inh_weight_ea,
            dout_eb => inh_weight_eb,
            dout_ec => inh_weight_ec,
            dout_ed => inh_weight_ed,
            dout_ee => inh_weight_ee,
            dout_ef => inh_weight_ef,
            dout_f0 => inh_weight_f0,
            dout_f1 => inh_weight_f1,
            dout_f2 => inh_weight_f2,
            dout_f3 => inh_weight_f3,
            dout_f4 => inh_weight_f4,
            dout_f5 => inh_weight_f5,
            dout_f6 => inh_weight_f6,
            dout_f7 => inh_weight_f7,
            dout_f8 => inh_weight_f8,
            dout_f9 => inh_weight_f9,
            dout_fa => inh_weight_fa,
            dout_fb => inh_weight_fb,
            dout_fc => inh_weight_fc,
            dout_fd => inh_weight_fd,
            dout_fe => inh_weight_fe,
            dout_ff => inh_weight_ff,
            dout_100 => inh_weight_100,
            dout_101 => inh_weight_101,
            dout_102 => inh_weight_102,
            dout_103 => inh_weight_103,
            dout_104 => inh_weight_104,
            dout_105 => inh_weight_105,
            dout_106 => inh_weight_106,
            dout_107 => inh_weight_107,
            dout_108 => inh_weight_108,
            dout_109 => inh_weight_109,
            dout_10a => inh_weight_10a,
            dout_10b => inh_weight_10b,
            dout_10c => inh_weight_10c,
            dout_10d => inh_weight_10d,
            dout_10e => inh_weight_10e,
            dout_10f => inh_weight_10f,
            dout_110 => inh_weight_110,
            dout_111 => inh_weight_111,
            dout_112 => inh_weight_112,
            dout_113 => inh_weight_113,
            dout_114 => inh_weight_114,
            dout_115 => inh_weight_115,
            dout_116 => inh_weight_116,
            dout_117 => inh_weight_117,
            dout_118 => inh_weight_118,
            dout_119 => inh_weight_119,
            dout_11a => inh_weight_11a,
            dout_11b => inh_weight_11b,
            dout_11c => inh_weight_11c,
            dout_11d => inh_weight_11d,
            dout_11e => inh_weight_11e,
            dout_11f => inh_weight_11f,
            dout_120 => inh_weight_120,
            dout_121 => inh_weight_121,
            dout_122 => inh_weight_122,
            dout_123 => inh_weight_123,
            dout_124 => inh_weight_124,
            dout_125 => inh_weight_125,
            dout_126 => inh_weight_126,
            dout_127 => inh_weight_127,
            dout_128 => inh_weight_128,
            dout_129 => inh_weight_129,
            dout_12a => inh_weight_12a,
            dout_12b => inh_weight_12b,
            dout_12c => inh_weight_12c,
            dout_12d => inh_weight_12d,
            dout_12e => inh_weight_12e,
            dout_12f => inh_weight_12f,
            dout_130 => inh_weight_130,
            dout_131 => inh_weight_131,
            dout_132 => inh_weight_132,
            dout_133 => inh_weight_133,
            dout_134 => inh_weight_134,
            dout_135 => inh_weight_135,
            dout_136 => inh_weight_136,
            dout_137 => inh_weight_137,
            dout_138 => inh_weight_138,
            dout_139 => inh_weight_139,
            dout_13a => inh_weight_13a,
            dout_13b => inh_weight_13b,
            dout_13c => inh_weight_13c,
            dout_13d => inh_weight_13d,
            dout_13e => inh_weight_13e,
            dout_13f => inh_weight_13f,
            dout_140 => inh_weight_140,
            dout_141 => inh_weight_141,
            dout_142 => inh_weight_142,
            dout_143 => inh_weight_143,
            dout_144 => inh_weight_144,
            dout_145 => inh_weight_145,
            dout_146 => inh_weight_146,
            dout_147 => inh_weight_147,
            dout_148 => inh_weight_148,
            dout_149 => inh_weight_149,
            dout_14a => inh_weight_14a,
            dout_14b => inh_weight_14b,
            dout_14c => inh_weight_14c,
            dout_14d => inh_weight_14d,
            dout_14e => inh_weight_14e,
            dout_14f => inh_weight_14f,
            dout_150 => inh_weight_150,
            dout_151 => inh_weight_151,
            dout_152 => inh_weight_152,
            dout_153 => inh_weight_153,
            dout_154 => inh_weight_154,
            dout_155 => inh_weight_155,
            dout_156 => inh_weight_156,
            dout_157 => inh_weight_157,
            dout_158 => inh_weight_158,
            dout_159 => inh_weight_159,
            dout_15a => inh_weight_15a,
            dout_15b => inh_weight_15b,
            dout_15c => inh_weight_15c,
            dout_15d => inh_weight_15d,
            dout_15e => inh_weight_15e,
            dout_15f => inh_weight_15f,
            dout_160 => inh_weight_160,
            dout_161 => inh_weight_161,
            dout_162 => inh_weight_162,
            dout_163 => inh_weight_163,
            dout_164 => inh_weight_164,
            dout_165 => inh_weight_165,
            dout_166 => inh_weight_166,
            dout_167 => inh_weight_167,
            dout_168 => inh_weight_168,
            dout_169 => inh_weight_169,
            dout_16a => inh_weight_16a,
            dout_16b => inh_weight_16b,
            dout_16c => inh_weight_16c,
            dout_16d => inh_weight_16d,
            dout_16e => inh_weight_16e,
            dout_16f => inh_weight_16f,
            dout_170 => inh_weight_170,
            dout_171 => inh_weight_171,
            dout_172 => inh_weight_172,
            dout_173 => inh_weight_173,
            dout_174 => inh_weight_174,
            dout_175 => inh_weight_175,
            dout_176 => inh_weight_176,
            dout_177 => inh_weight_177,
            dout_178 => inh_weight_178,
            dout_179 => inh_weight_179,
            dout_17a => inh_weight_17a,
            dout_17b => inh_weight_17b,
            dout_17c => inh_weight_17c,
            dout_17d => inh_weight_17d,
            dout_17e => inh_weight_17e,
            dout_17f => inh_weight_17f,
            dout_180 => inh_weight_180,
            dout_181 => inh_weight_181,
            dout_182 => inh_weight_182,
            dout_183 => inh_weight_183,
            dout_184 => inh_weight_184,
            dout_185 => inh_weight_185,
            dout_186 => inh_weight_186,
            dout_187 => inh_weight_187,
            dout_188 => inh_weight_188,
            dout_189 => inh_weight_189,
            dout_18a => inh_weight_18a,
            dout_18b => inh_weight_18b,
            dout_18c => inh_weight_18c,
            dout_18d => inh_weight_18d,
            dout_18e => inh_weight_18e,
            dout_18f => inh_weight_18f,
            dout_190 => inh_weight_190,
            dout_191 => inh_weight_191,
            dout_192 => inh_weight_192,
            dout_193 => inh_weight_193,
            dout_194 => inh_weight_194,
            dout_195 => inh_weight_195,
            dout_196 => inh_weight_196,
            dout_197 => inh_weight_197,
            dout_198 => inh_weight_198,
            dout_199 => inh_weight_199,
            dout_19a => inh_weight_19a,
            dout_19b => inh_weight_19b,
            dout_19c => inh_weight_19c,
            dout_19d => inh_weight_19d,
            dout_19e => inh_weight_19e,
            dout_19f => inh_weight_19f,
            dout_1a0 => inh_weight_1a0,
            dout_1a1 => inh_weight_1a1,
            dout_1a2 => inh_weight_1a2,
            dout_1a3 => inh_weight_1a3,
            dout_1a4 => inh_weight_1a4,
            dout_1a5 => inh_weight_1a5,
            dout_1a6 => inh_weight_1a6,
            dout_1a7 => inh_weight_1a7,
            dout_1a8 => inh_weight_1a8,
            dout_1a9 => inh_weight_1a9,
            dout_1aa => inh_weight_1aa,
            dout_1ab => inh_weight_1ab,
            dout_1ac => inh_weight_1ac,
            dout_1ad => inh_weight_1ad,
            dout_1ae => inh_weight_1ae,
            dout_1af => inh_weight_1af,
            dout_1b0 => inh_weight_1b0,
            dout_1b1 => inh_weight_1b1,
            dout_1b2 => inh_weight_1b2,
            dout_1b3 => inh_weight_1b3,
            dout_1b4 => inh_weight_1b4,
            dout_1b5 => inh_weight_1b5,
            dout_1b6 => inh_weight_1b6,
            dout_1b7 => inh_weight_1b7,
            dout_1b8 => inh_weight_1b8,
            dout_1b9 => inh_weight_1b9,
            dout_1ba => inh_weight_1ba,
            dout_1bb => inh_weight_1bb,
            dout_1bc => inh_weight_1bc,
            dout_1bd => inh_weight_1bd,
            dout_1be => inh_weight_1be,
            dout_1bf => inh_weight_1bf,
            dout_1c0 => inh_weight_1c0,
            dout_1c1 => inh_weight_1c1,
            dout_1c2 => inh_weight_1c2,
            dout_1c3 => inh_weight_1c3,
            dout_1c4 => inh_weight_1c4,
            dout_1c5 => inh_weight_1c5,
            dout_1c6 => inh_weight_1c6,
            dout_1c7 => inh_weight_1c7,
            dout_1c8 => inh_weight_1c8,
            dout_1c9 => inh_weight_1c9,
            dout_1ca => inh_weight_1ca,
            dout_1cb => inh_weight_1cb,
            dout_1cc => inh_weight_1cc,
            dout_1cd => inh_weight_1cd,
            dout_1ce => inh_weight_1ce,
            dout_1cf => inh_weight_1cf,
            dout_1d0 => inh_weight_1d0,
            dout_1d1 => inh_weight_1d1,
            dout_1d2 => inh_weight_1d2,
            dout_1d3 => inh_weight_1d3,
            dout_1d4 => inh_weight_1d4,
            dout_1d5 => inh_weight_1d5,
            dout_1d6 => inh_weight_1d6,
            dout_1d7 => inh_weight_1d7,
            dout_1d8 => inh_weight_1d8,
            dout_1d9 => inh_weight_1d9,
            dout_1da => inh_weight_1da,
            dout_1db => inh_weight_1db,
            dout_1dc => inh_weight_1dc,
            dout_1dd => inh_weight_1dd,
            dout_1de => inh_weight_1de,
            dout_1df => inh_weight_1df,
            dout_1e0 => inh_weight_1e0,
            dout_1e1 => inh_weight_1e1,
            dout_1e2 => inh_weight_1e2,
            dout_1e3 => inh_weight_1e3,
            dout_1e4 => inh_weight_1e4,
            dout_1e5 => inh_weight_1e5,
            dout_1e6 => inh_weight_1e6,
            dout_1e7 => inh_weight_1e7,
            dout_1e8 => inh_weight_1e8,
            dout_1e9 => inh_weight_1e9,
            dout_1ea => inh_weight_1ea,
            dout_1eb => inh_weight_1eb,
            dout_1ec => inh_weight_1ec,
            dout_1ed => inh_weight_1ed,
            dout_1ee => inh_weight_1ee,
            dout_1ef => inh_weight_1ef,
            dout_1f0 => inh_weight_1f0,
            dout_1f1 => inh_weight_1f1,
            dout_1f2 => inh_weight_1f2,
            dout_1f3 => inh_weight_1f3,
            dout_1f4 => inh_weight_1f4,
            dout_1f5 => inh_weight_1f5,
            dout_1f6 => inh_weight_1f6,
            dout_1f7 => inh_weight_1f7,
            dout_1f8 => inh_weight_1f8,
            dout_1f9 => inh_weight_1f9,
            dout_1fa => inh_weight_1fa,
            dout_1fb => inh_weight_1fb,
            dout_1fc => inh_weight_1fc,
            dout_1fd => inh_weight_1fd,
            dout_1fe => inh_weight_1fe,
            dout_1ff => inh_weight_1ff
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
            N => 512
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

