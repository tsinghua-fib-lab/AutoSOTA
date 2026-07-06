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


entity multi_input_dp_128_exc_10_inh is
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
end entity multi_input_dp_128_exc_10_inh;

architecture behavior of multi_input_dp_128_exc_10_inh is


    component generic_or is
        generic (
            N : integer := 128
        );
        port (
            or_in : in std_logic_vector(N-1 downto 0);
            or_out : out std_logic
        );
    end component;

    component reg_sync_rst is
        generic (
            N : integer := 128
        );
        port (
            clk : in std_logic;
            en : in std_logic;
            rst_n : in std_logic;
            reg_in : in std_logic_vector(N-1 downto 0);
            reg_out : out std_logic_vector(N-1 downto 0)
        );
    end component;

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
            N : integer := 7
        );
        port (
            in0 : in std_logic_vector(N-1 downto 0);
            in1 : in std_logic_vector(N-1 downto 0);
            cmp_out : out std_logic
        );
    end component;

    component mux_128to1 is
        port (
            mux_sel : in std_logic_vector(6 downto 0);
            in0 : in std_logic;
            in1 : in std_logic;
            in2 : in std_logic;
            in3 : in std_logic;
            in4 : in std_logic;
            in5 : in std_logic;
            in6 : in std_logic;
            in7 : in std_logic;
            in8 : in std_logic;
            in9 : in std_logic;
            in10 : in std_logic;
            in11 : in std_logic;
            in12 : in std_logic;
            in13 : in std_logic;
            in14 : in std_logic;
            in15 : in std_logic;
            in16 : in std_logic;
            in17 : in std_logic;
            in18 : in std_logic;
            in19 : in std_logic;
            in20 : in std_logic;
            in21 : in std_logic;
            in22 : in std_logic;
            in23 : in std_logic;
            in24 : in std_logic;
            in25 : in std_logic;
            in26 : in std_logic;
            in27 : in std_logic;
            in28 : in std_logic;
            in29 : in std_logic;
            in30 : in std_logic;
            in31 : in std_logic;
            in32 : in std_logic;
            in33 : in std_logic;
            in34 : in std_logic;
            in35 : in std_logic;
            in36 : in std_logic;
            in37 : in std_logic;
            in38 : in std_logic;
            in39 : in std_logic;
            in40 : in std_logic;
            in41 : in std_logic;
            in42 : in std_logic;
            in43 : in std_logic;
            in44 : in std_logic;
            in45 : in std_logic;
            in46 : in std_logic;
            in47 : in std_logic;
            in48 : in std_logic;
            in49 : in std_logic;
            in50 : in std_logic;
            in51 : in std_logic;
            in52 : in std_logic;
            in53 : in std_logic;
            in54 : in std_logic;
            in55 : in std_logic;
            in56 : in std_logic;
            in57 : in std_logic;
            in58 : in std_logic;
            in59 : in std_logic;
            in60 : in std_logic;
            in61 : in std_logic;
            in62 : in std_logic;
            in63 : in std_logic;
            in64 : in std_logic;
            in65 : in std_logic;
            in66 : in std_logic;
            in67 : in std_logic;
            in68 : in std_logic;
            in69 : in std_logic;
            in70 : in std_logic;
            in71 : in std_logic;
            in72 : in std_logic;
            in73 : in std_logic;
            in74 : in std_logic;
            in75 : in std_logic;
            in76 : in std_logic;
            in77 : in std_logic;
            in78 : in std_logic;
            in79 : in std_logic;
            in80 : in std_logic;
            in81 : in std_logic;
            in82 : in std_logic;
            in83 : in std_logic;
            in84 : in std_logic;
            in85 : in std_logic;
            in86 : in std_logic;
            in87 : in std_logic;
            in88 : in std_logic;
            in89 : in std_logic;
            in90 : in std_logic;
            in91 : in std_logic;
            in92 : in std_logic;
            in93 : in std_logic;
            in94 : in std_logic;
            in95 : in std_logic;
            in96 : in std_logic;
            in97 : in std_logic;
            in98 : in std_logic;
            in99 : in std_logic;
            in100 : in std_logic;
            in101 : in std_logic;
            in102 : in std_logic;
            in103 : in std_logic;
            in104 : in std_logic;
            in105 : in std_logic;
            in106 : in std_logic;
            in107 : in std_logic;
            in108 : in std_logic;
            in109 : in std_logic;
            in110 : in std_logic;
            in111 : in std_logic;
            in112 : in std_logic;
            in113 : in std_logic;
            in114 : in std_logic;
            in115 : in std_logic;
            in116 : in std_logic;
            in117 : in std_logic;
            in118 : in std_logic;
            in119 : in std_logic;
            in120 : in std_logic;
            in121 : in std_logic;
            in122 : in std_logic;
            in123 : in std_logic;
            in124 : in std_logic;
            in125 : in std_logic;
            in126 : in std_logic;
            in127 : in std_logic;
            mux_out : out std_logic
        );
    end component;

    component mux_16to1 is
        port (
            mux_sel : in std_logic_vector(3 downto 0);
            in0 : in std_logic;
            in1 : in std_logic;
            in2 : in std_logic;
            in3 : in std_logic;
            in4 : in std_logic;
            in5 : in std_logic;
            in6 : in std_logic;
            in7 : in std_logic;
            in8 : in std_logic;
            in9 : in std_logic;
            in10 : in std_logic;
            in11 : in std_logic;
            in12 : in std_logic;
            in13 : in std_logic;
            in14 : in std_logic;
            in15 : in std_logic;
            mux_out : out std_logic
        );
    end component;


    signal exc_spikes_sampled : std_logic_vector(n_exc_inputs-1 downto 0);
    signal inh_spikes_sampled : std_logic_vector(n_inh_inputs-1 downto 0);
    signal exc_cnt_sig : std_logic_vector(exc_cnt_bitwidth-1 downto 0);
    signal inh_cnt_sig : std_logic_vector(inh_cnt_bitwidth-1 downto 0);

begin

    exc_cnt <= exc_cnt_sig;
    inh_cnt <= inh_cnt_sig;


    exc_or : generic_or
        generic map(
            N => n_exc_inputs
        )
        port map(
            or_in => exc_spikes,
            or_out => exc_yes
        );

    inh_or : generic_or
        generic map(
            N => n_inh_inputs
        )
        port map(
            or_in => inh_spikes,
            or_out => inh_yes
        );

    exc_reg : reg_sync_rst
        generic map(
            N => n_exc_inputs
        )
        port map(
            clk => clk,
            en => exc_sample,
            rst_n => exc_rst_n,
            reg_in => exc_spikes,
            reg_out => exc_spikes_sampled
        );

    inh_reg : reg_sync_rst
        generic map(
            N => n_inh_inputs
        )
        port map(
            clk => clk,
            en => inh_sample,
            rst_n => inh_rst_n,
            reg_in => inh_spikes,
            reg_out => inh_spikes_sampled
        );

    exc_mux : mux_128to1
        port map(
            mux_sel => exc_cnt_sig,
            in0 => exc_spikes_sampled(0),
            in1 => exc_spikes_sampled(1),
            in2 => exc_spikes_sampled(2),
            in3 => exc_spikes_sampled(3),
            in4 => exc_spikes_sampled(4),
            in5 => exc_spikes_sampled(5),
            in6 => exc_spikes_sampled(6),
            in7 => exc_spikes_sampled(7),
            in8 => exc_spikes_sampled(8),
            in9 => exc_spikes_sampled(9),
            in10 => exc_spikes_sampled(10),
            in11 => exc_spikes_sampled(11),
            in12 => exc_spikes_sampled(12),
            in13 => exc_spikes_sampled(13),
            in14 => exc_spikes_sampled(14),
            in15 => exc_spikes_sampled(15),
            in16 => exc_spikes_sampled(16),
            in17 => exc_spikes_sampled(17),
            in18 => exc_spikes_sampled(18),
            in19 => exc_spikes_sampled(19),
            in20 => exc_spikes_sampled(20),
            in21 => exc_spikes_sampled(21),
            in22 => exc_spikes_sampled(22),
            in23 => exc_spikes_sampled(23),
            in24 => exc_spikes_sampled(24),
            in25 => exc_spikes_sampled(25),
            in26 => exc_spikes_sampled(26),
            in27 => exc_spikes_sampled(27),
            in28 => exc_spikes_sampled(28),
            in29 => exc_spikes_sampled(29),
            in30 => exc_spikes_sampled(30),
            in31 => exc_spikes_sampled(31),
            in32 => exc_spikes_sampled(32),
            in33 => exc_spikes_sampled(33),
            in34 => exc_spikes_sampled(34),
            in35 => exc_spikes_sampled(35),
            in36 => exc_spikes_sampled(36),
            in37 => exc_spikes_sampled(37),
            in38 => exc_spikes_sampled(38),
            in39 => exc_spikes_sampled(39),
            in40 => exc_spikes_sampled(40),
            in41 => exc_spikes_sampled(41),
            in42 => exc_spikes_sampled(42),
            in43 => exc_spikes_sampled(43),
            in44 => exc_spikes_sampled(44),
            in45 => exc_spikes_sampled(45),
            in46 => exc_spikes_sampled(46),
            in47 => exc_spikes_sampled(47),
            in48 => exc_spikes_sampled(48),
            in49 => exc_spikes_sampled(49),
            in50 => exc_spikes_sampled(50),
            in51 => exc_spikes_sampled(51),
            in52 => exc_spikes_sampled(52),
            in53 => exc_spikes_sampled(53),
            in54 => exc_spikes_sampled(54),
            in55 => exc_spikes_sampled(55),
            in56 => exc_spikes_sampled(56),
            in57 => exc_spikes_sampled(57),
            in58 => exc_spikes_sampled(58),
            in59 => exc_spikes_sampled(59),
            in60 => exc_spikes_sampled(60),
            in61 => exc_spikes_sampled(61),
            in62 => exc_spikes_sampled(62),
            in63 => exc_spikes_sampled(63),
            in64 => exc_spikes_sampled(64),
            in65 => exc_spikes_sampled(65),
            in66 => exc_spikes_sampled(66),
            in67 => exc_spikes_sampled(67),
            in68 => exc_spikes_sampled(68),
            in69 => exc_spikes_sampled(69),
            in70 => exc_spikes_sampled(70),
            in71 => exc_spikes_sampled(71),
            in72 => exc_spikes_sampled(72),
            in73 => exc_spikes_sampled(73),
            in74 => exc_spikes_sampled(74),
            in75 => exc_spikes_sampled(75),
            in76 => exc_spikes_sampled(76),
            in77 => exc_spikes_sampled(77),
            in78 => exc_spikes_sampled(78),
            in79 => exc_spikes_sampled(79),
            in80 => exc_spikes_sampled(80),
            in81 => exc_spikes_sampled(81),
            in82 => exc_spikes_sampled(82),
            in83 => exc_spikes_sampled(83),
            in84 => exc_spikes_sampled(84),
            in85 => exc_spikes_sampled(85),
            in86 => exc_spikes_sampled(86),
            in87 => exc_spikes_sampled(87),
            in88 => exc_spikes_sampled(88),
            in89 => exc_spikes_sampled(89),
            in90 => exc_spikes_sampled(90),
            in91 => exc_spikes_sampled(91),
            in92 => exc_spikes_sampled(92),
            in93 => exc_spikes_sampled(93),
            in94 => exc_spikes_sampled(94),
            in95 => exc_spikes_sampled(95),
            in96 => exc_spikes_sampled(96),
            in97 => exc_spikes_sampled(97),
            in98 => exc_spikes_sampled(98),
            in99 => exc_spikes_sampled(99),
            in100 => exc_spikes_sampled(100),
            in101 => exc_spikes_sampled(101),
            in102 => exc_spikes_sampled(102),
            in103 => exc_spikes_sampled(103),
            in104 => exc_spikes_sampled(104),
            in105 => exc_spikes_sampled(105),
            in106 => exc_spikes_sampled(106),
            in107 => exc_spikes_sampled(107),
            in108 => exc_spikes_sampled(108),
            in109 => exc_spikes_sampled(109),
            in110 => exc_spikes_sampled(110),
            in111 => exc_spikes_sampled(111),
            in112 => exc_spikes_sampled(112),
            in113 => exc_spikes_sampled(113),
            in114 => exc_spikes_sampled(114),
            in115 => exc_spikes_sampled(115),
            in116 => exc_spikes_sampled(116),
            in117 => exc_spikes_sampled(117),
            in118 => exc_spikes_sampled(118),
            in119 => exc_spikes_sampled(119),
            in120 => exc_spikes_sampled(120),
            in121 => exc_spikes_sampled(121),
            in122 => exc_spikes_sampled(122),
            in123 => exc_spikes_sampled(123),
            in124 => exc_spikes_sampled(124),
            in125 => exc_spikes_sampled(125),
            in126 => exc_spikes_sampled(126),
            in127 => exc_spikes_sampled(127),
            mux_out => exc_spike
        );

    inh_mux : mux_16to1
        port map(
            mux_sel => inh_cnt_sig,
            in0 => inh_spikes_sampled(0),
            in1 => inh_spikes_sampled(1),
            in2 => inh_spikes_sampled(2),
            in3 => inh_spikes_sampled(3),
            in4 => inh_spikes_sampled(4),
            in5 => inh_spikes_sampled(5),
            in6 => inh_spikes_sampled(6),
            in7 => inh_spikes_sampled(7),
            in8 => inh_spikes_sampled(8),
            in9 => inh_spikes_sampled(9),
            in10 => '0',
            in11 => '0',
            in12 => '0',
            in13 => '0',
            in14 => '0',
            in15 => '0',
            mux_out => inh_spike
        );

    exc_counter : cnt
        generic map(
            N => exc_cnt_bitwidth,
            rst_value => 127
        )
        port map(
            clk => clk,
            cnt_en => exc_cnt_en,
            cnt_rst_n => exc_cnt_rst_n,
            cnt_out => exc_cnt_sig
        );

    inh_counter : cnt
        generic map(
            N => inh_cnt_bitwidth,
            rst_value => 15
        )
        port map(
            clk => clk,
            cnt_en => inh_cnt_en,
            cnt_rst_n => inh_cnt_rst_n,
            cnt_out => inh_cnt_sig
        );

    exc_cmp : cmp_eq
        generic map(
            N => exc_cnt_bitwidth
        )
        port map(
            in0 => exc_cnt_sig,
            in1 => std_logic_vector(to_unsigned(n_exc_inputs-2, exc_cnt_bitwidth)),
            cmp_out => exc_stop
        );

    inh_cmp : cmp_eq
        generic map(
            N => inh_cnt_bitwidth
        )
        port map(
            in0 => inh_cnt_sig,
            in1 => std_logic_vector(to_unsigned(n_inh_inputs-2, inh_cnt_bitwidth)),
            cmp_out => inh_stop
        );


end architecture behavior;

