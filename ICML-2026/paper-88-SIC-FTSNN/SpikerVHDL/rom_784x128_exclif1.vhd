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


entity rom_784x128_exclif1 is
    port (
        clka : in std_logic;
        addra : in std_logic_vector(9 downto 0);
        dout_0 : out std_logic_vector(7 downto 0);
        dout_1 : out std_logic_vector(7 downto 0);
        dout_2 : out std_logic_vector(7 downto 0);
        dout_3 : out std_logic_vector(7 downto 0);
        dout_4 : out std_logic_vector(7 downto 0);
        dout_5 : out std_logic_vector(7 downto 0);
        dout_6 : out std_logic_vector(7 downto 0);
        dout_7 : out std_logic_vector(7 downto 0);
        dout_8 : out std_logic_vector(7 downto 0);
        dout_9 : out std_logic_vector(7 downto 0);
        dout_a : out std_logic_vector(7 downto 0);
        dout_b : out std_logic_vector(7 downto 0);
        dout_c : out std_logic_vector(7 downto 0);
        dout_d : out std_logic_vector(7 downto 0);
        dout_e : out std_logic_vector(7 downto 0);
        dout_f : out std_logic_vector(7 downto 0);
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
        dout_7f : out std_logic_vector(7 downto 0)
    );
end entity rom_784x128_exclif1;

architecture behavior of rom_784x128_exclif1 is


    component rom_784x128_exclif1_ip is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(9 downto 0);
            douta : out std_logic_vector(1023 downto 0)
        );
    end component;


    signal douta : std_logic_vector(1023 downto 0);

begin

    dout_0 <= douta(7 downto 0);
    dout_1 <= douta(15 downto 8);
    dout_2 <= douta(23 downto 16);
    dout_3 <= douta(31 downto 24);
    dout_4 <= douta(39 downto 32);
    dout_5 <= douta(47 downto 40);
    dout_6 <= douta(55 downto 48);
    dout_7 <= douta(63 downto 56);
    dout_8 <= douta(71 downto 64);
    dout_9 <= douta(79 downto 72);
    dout_a <= douta(87 downto 80);
    dout_b <= douta(95 downto 88);
    dout_c <= douta(103 downto 96);
    dout_d <= douta(111 downto 104);
    dout_e <= douta(119 downto 112);
    dout_f <= douta(127 downto 120);
    dout_10 <= douta(135 downto 128);
    dout_11 <= douta(143 downto 136);
    dout_12 <= douta(151 downto 144);
    dout_13 <= douta(159 downto 152);
    dout_14 <= douta(167 downto 160);
    dout_15 <= douta(175 downto 168);
    dout_16 <= douta(183 downto 176);
    dout_17 <= douta(191 downto 184);
    dout_18 <= douta(199 downto 192);
    dout_19 <= douta(207 downto 200);
    dout_1a <= douta(215 downto 208);
    dout_1b <= douta(223 downto 216);
    dout_1c <= douta(231 downto 224);
    dout_1d <= douta(239 downto 232);
    dout_1e <= douta(247 downto 240);
    dout_1f <= douta(255 downto 248);
    dout_20 <= douta(263 downto 256);
    dout_21 <= douta(271 downto 264);
    dout_22 <= douta(279 downto 272);
    dout_23 <= douta(287 downto 280);
    dout_24 <= douta(295 downto 288);
    dout_25 <= douta(303 downto 296);
    dout_26 <= douta(311 downto 304);
    dout_27 <= douta(319 downto 312);
    dout_28 <= douta(327 downto 320);
    dout_29 <= douta(335 downto 328);
    dout_2a <= douta(343 downto 336);
    dout_2b <= douta(351 downto 344);
    dout_2c <= douta(359 downto 352);
    dout_2d <= douta(367 downto 360);
    dout_2e <= douta(375 downto 368);
    dout_2f <= douta(383 downto 376);
    dout_30 <= douta(391 downto 384);
    dout_31 <= douta(399 downto 392);
    dout_32 <= douta(407 downto 400);
    dout_33 <= douta(415 downto 408);
    dout_34 <= douta(423 downto 416);
    dout_35 <= douta(431 downto 424);
    dout_36 <= douta(439 downto 432);
    dout_37 <= douta(447 downto 440);
    dout_38 <= douta(455 downto 448);
    dout_39 <= douta(463 downto 456);
    dout_3a <= douta(471 downto 464);
    dout_3b <= douta(479 downto 472);
    dout_3c <= douta(487 downto 480);
    dout_3d <= douta(495 downto 488);
    dout_3e <= douta(503 downto 496);
    dout_3f <= douta(511 downto 504);
    dout_40 <= douta(519 downto 512);
    dout_41 <= douta(527 downto 520);
    dout_42 <= douta(535 downto 528);
    dout_43 <= douta(543 downto 536);
    dout_44 <= douta(551 downto 544);
    dout_45 <= douta(559 downto 552);
    dout_46 <= douta(567 downto 560);
    dout_47 <= douta(575 downto 568);
    dout_48 <= douta(583 downto 576);
    dout_49 <= douta(591 downto 584);
    dout_4a <= douta(599 downto 592);
    dout_4b <= douta(607 downto 600);
    dout_4c <= douta(615 downto 608);
    dout_4d <= douta(623 downto 616);
    dout_4e <= douta(631 downto 624);
    dout_4f <= douta(639 downto 632);
    dout_50 <= douta(647 downto 640);
    dout_51 <= douta(655 downto 648);
    dout_52 <= douta(663 downto 656);
    dout_53 <= douta(671 downto 664);
    dout_54 <= douta(679 downto 672);
    dout_55 <= douta(687 downto 680);
    dout_56 <= douta(695 downto 688);
    dout_57 <= douta(703 downto 696);
    dout_58 <= douta(711 downto 704);
    dout_59 <= douta(719 downto 712);
    dout_5a <= douta(727 downto 720);
    dout_5b <= douta(735 downto 728);
    dout_5c <= douta(743 downto 736);
    dout_5d <= douta(751 downto 744);
    dout_5e <= douta(759 downto 752);
    dout_5f <= douta(767 downto 760);
    dout_60 <= douta(775 downto 768);
    dout_61 <= douta(783 downto 776);
    dout_62 <= douta(791 downto 784);
    dout_63 <= douta(799 downto 792);
    dout_64 <= douta(807 downto 800);
    dout_65 <= douta(815 downto 808);
    dout_66 <= douta(823 downto 816);
    dout_67 <= douta(831 downto 824);
    dout_68 <= douta(839 downto 832);
    dout_69 <= douta(847 downto 840);
    dout_6a <= douta(855 downto 848);
    dout_6b <= douta(863 downto 856);
    dout_6c <= douta(871 downto 864);
    dout_6d <= douta(879 downto 872);
    dout_6e <= douta(887 downto 880);
    dout_6f <= douta(895 downto 888);
    dout_70 <= douta(903 downto 896);
    dout_71 <= douta(911 downto 904);
    dout_72 <= douta(919 downto 912);
    dout_73 <= douta(927 downto 920);
    dout_74 <= douta(935 downto 928);
    dout_75 <= douta(943 downto 936);
    dout_76 <= douta(951 downto 944);
    dout_77 <= douta(959 downto 952);
    dout_78 <= douta(967 downto 960);
    dout_79 <= douta(975 downto 968);
    dout_7a <= douta(983 downto 976);
    dout_7b <= douta(991 downto 984);
    dout_7c <= douta(999 downto 992);
    dout_7d <= douta(1007 downto 1000);
    dout_7e <= douta(1015 downto 1008);
    dout_7f <= douta(1023 downto 1016);


    rom_784x128_exclif1_ip_instance : rom_784x128_exclif1_ip
        port map(
            clka => clka,
            addra => addra,
            douta => douta
        );


end architecture behavior;

