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


entity mux_128to1 is
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
end entity mux_128to1;

architecture behavior of mux_128to1 is


begin

    selection : process(mux_sel, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19, in20, in21, in22, in23, in24, in25, in26, in27, in28, in29, in30, in31, in32, in33, in34, in35, in36, in37, in38, in39, in40, in41, in42, in43, in44, in45, in46, in47, in48, in49, in50, in51, in52, in53, in54, in55, in56, in57, in58, in59, in60, in61, in62, in63, in64, in65, in66, in67, in68, in69, in70, in71, in72, in73, in74, in75, in76, in77, in78, in79, in80, in81, in82, in83, in84, in85, in86, in87, in88, in89, in90, in91, in92, in93, in94, in95, in96, in97, in98, in99, in100, in101, in102, in103, in104, in105, in106, in107, in108, in109, in110, in111, in112, in113, in114, in115, in116, in117, in118, in119, in120, in121, in122, in123, in124, in125, in126, in127 )
    begin

        case mux_sel is

            when "0000000" =>
                mux_out <= in0;


            when "0000001" =>
                mux_out <= in1;


            when "0000010" =>
                mux_out <= in2;


            when "0000011" =>
                mux_out <= in3;


            when "0000100" =>
                mux_out <= in4;


            when "0000101" =>
                mux_out <= in5;


            when "0000110" =>
                mux_out <= in6;


            when "0000111" =>
                mux_out <= in7;


            when "0001000" =>
                mux_out <= in8;


            when "0001001" =>
                mux_out <= in9;


            when "0001010" =>
                mux_out <= in10;


            when "0001011" =>
                mux_out <= in11;


            when "0001100" =>
                mux_out <= in12;


            when "0001101" =>
                mux_out <= in13;


            when "0001110" =>
                mux_out <= in14;


            when "0001111" =>
                mux_out <= in15;


            when "0010000" =>
                mux_out <= in16;


            when "0010001" =>
                mux_out <= in17;


            when "0010010" =>
                mux_out <= in18;


            when "0010011" =>
                mux_out <= in19;


            when "0010100" =>
                mux_out <= in20;


            when "0010101" =>
                mux_out <= in21;


            when "0010110" =>
                mux_out <= in22;


            when "0010111" =>
                mux_out <= in23;


            when "0011000" =>
                mux_out <= in24;


            when "0011001" =>
                mux_out <= in25;


            when "0011010" =>
                mux_out <= in26;


            when "0011011" =>
                mux_out <= in27;


            when "0011100" =>
                mux_out <= in28;


            when "0011101" =>
                mux_out <= in29;


            when "0011110" =>
                mux_out <= in30;


            when "0011111" =>
                mux_out <= in31;


            when "0100000" =>
                mux_out <= in32;


            when "0100001" =>
                mux_out <= in33;


            when "0100010" =>
                mux_out <= in34;


            when "0100011" =>
                mux_out <= in35;


            when "0100100" =>
                mux_out <= in36;


            when "0100101" =>
                mux_out <= in37;


            when "0100110" =>
                mux_out <= in38;


            when "0100111" =>
                mux_out <= in39;


            when "0101000" =>
                mux_out <= in40;


            when "0101001" =>
                mux_out <= in41;


            when "0101010" =>
                mux_out <= in42;


            when "0101011" =>
                mux_out <= in43;


            when "0101100" =>
                mux_out <= in44;


            when "0101101" =>
                mux_out <= in45;


            when "0101110" =>
                mux_out <= in46;


            when "0101111" =>
                mux_out <= in47;


            when "0110000" =>
                mux_out <= in48;


            when "0110001" =>
                mux_out <= in49;


            when "0110010" =>
                mux_out <= in50;


            when "0110011" =>
                mux_out <= in51;


            when "0110100" =>
                mux_out <= in52;


            when "0110101" =>
                mux_out <= in53;


            when "0110110" =>
                mux_out <= in54;


            when "0110111" =>
                mux_out <= in55;


            when "0111000" =>
                mux_out <= in56;


            when "0111001" =>
                mux_out <= in57;


            when "0111010" =>
                mux_out <= in58;


            when "0111011" =>
                mux_out <= in59;


            when "0111100" =>
                mux_out <= in60;


            when "0111101" =>
                mux_out <= in61;


            when "0111110" =>
                mux_out <= in62;


            when "0111111" =>
                mux_out <= in63;


            when "1000000" =>
                mux_out <= in64;


            when "1000001" =>
                mux_out <= in65;


            when "1000010" =>
                mux_out <= in66;


            when "1000011" =>
                mux_out <= in67;


            when "1000100" =>
                mux_out <= in68;


            when "1000101" =>
                mux_out <= in69;


            when "1000110" =>
                mux_out <= in70;


            when "1000111" =>
                mux_out <= in71;


            when "1001000" =>
                mux_out <= in72;


            when "1001001" =>
                mux_out <= in73;


            when "1001010" =>
                mux_out <= in74;


            when "1001011" =>
                mux_out <= in75;


            when "1001100" =>
                mux_out <= in76;


            when "1001101" =>
                mux_out <= in77;


            when "1001110" =>
                mux_out <= in78;


            when "1001111" =>
                mux_out <= in79;


            when "1010000" =>
                mux_out <= in80;


            when "1010001" =>
                mux_out <= in81;


            when "1010010" =>
                mux_out <= in82;


            when "1010011" =>
                mux_out <= in83;


            when "1010100" =>
                mux_out <= in84;


            when "1010101" =>
                mux_out <= in85;


            when "1010110" =>
                mux_out <= in86;


            when "1010111" =>
                mux_out <= in87;


            when "1011000" =>
                mux_out <= in88;


            when "1011001" =>
                mux_out <= in89;


            when "1011010" =>
                mux_out <= in90;


            when "1011011" =>
                mux_out <= in91;


            when "1011100" =>
                mux_out <= in92;


            when "1011101" =>
                mux_out <= in93;


            when "1011110" =>
                mux_out <= in94;


            when "1011111" =>
                mux_out <= in95;


            when "1100000" =>
                mux_out <= in96;


            when "1100001" =>
                mux_out <= in97;


            when "1100010" =>
                mux_out <= in98;


            when "1100011" =>
                mux_out <= in99;


            when "1100100" =>
                mux_out <= in100;


            when "1100101" =>
                mux_out <= in101;


            when "1100110" =>
                mux_out <= in102;


            when "1100111" =>
                mux_out <= in103;


            when "1101000" =>
                mux_out <= in104;


            when "1101001" =>
                mux_out <= in105;


            when "1101010" =>
                mux_out <= in106;


            when "1101011" =>
                mux_out <= in107;


            when "1101100" =>
                mux_out <= in108;


            when "1101101" =>
                mux_out <= in109;


            when "1101110" =>
                mux_out <= in110;


            when "1101111" =>
                mux_out <= in111;


            when "1110000" =>
                mux_out <= in112;


            when "1110001" =>
                mux_out <= in113;


            when "1110010" =>
                mux_out <= in114;


            when "1110011" =>
                mux_out <= in115;


            when "1110100" =>
                mux_out <= in116;


            when "1110101" =>
                mux_out <= in117;


            when "1110110" =>
                mux_out <= in118;


            when "1110111" =>
                mux_out <= in119;


            when "1111000" =>
                mux_out <= in120;


            when "1111001" =>
                mux_out <= in121;


            when "1111010" =>
                mux_out <= in122;


            when "1111011" =>
                mux_out <= in123;


            when "1111100" =>
                mux_out <= in124;


            when "1111101" =>
                mux_out <= in125;


            when "1111110" =>
                mux_out <= in126;


            when others =>
                mux_out <= in127;


        end case;

    end process selection;


end architecture behavior;

