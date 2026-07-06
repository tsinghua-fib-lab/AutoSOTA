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


entity mux_512to1 is
    port (
        mux_sel : in std_logic_vector(8 downto 0);
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
        in128 : in std_logic;
        in129 : in std_logic;
        in130 : in std_logic;
        in131 : in std_logic;
        in132 : in std_logic;
        in133 : in std_logic;
        in134 : in std_logic;
        in135 : in std_logic;
        in136 : in std_logic;
        in137 : in std_logic;
        in138 : in std_logic;
        in139 : in std_logic;
        in140 : in std_logic;
        in141 : in std_logic;
        in142 : in std_logic;
        in143 : in std_logic;
        in144 : in std_logic;
        in145 : in std_logic;
        in146 : in std_logic;
        in147 : in std_logic;
        in148 : in std_logic;
        in149 : in std_logic;
        in150 : in std_logic;
        in151 : in std_logic;
        in152 : in std_logic;
        in153 : in std_logic;
        in154 : in std_logic;
        in155 : in std_logic;
        in156 : in std_logic;
        in157 : in std_logic;
        in158 : in std_logic;
        in159 : in std_logic;
        in160 : in std_logic;
        in161 : in std_logic;
        in162 : in std_logic;
        in163 : in std_logic;
        in164 : in std_logic;
        in165 : in std_logic;
        in166 : in std_logic;
        in167 : in std_logic;
        in168 : in std_logic;
        in169 : in std_logic;
        in170 : in std_logic;
        in171 : in std_logic;
        in172 : in std_logic;
        in173 : in std_logic;
        in174 : in std_logic;
        in175 : in std_logic;
        in176 : in std_logic;
        in177 : in std_logic;
        in178 : in std_logic;
        in179 : in std_logic;
        in180 : in std_logic;
        in181 : in std_logic;
        in182 : in std_logic;
        in183 : in std_logic;
        in184 : in std_logic;
        in185 : in std_logic;
        in186 : in std_logic;
        in187 : in std_logic;
        in188 : in std_logic;
        in189 : in std_logic;
        in190 : in std_logic;
        in191 : in std_logic;
        in192 : in std_logic;
        in193 : in std_logic;
        in194 : in std_logic;
        in195 : in std_logic;
        in196 : in std_logic;
        in197 : in std_logic;
        in198 : in std_logic;
        in199 : in std_logic;
        in200 : in std_logic;
        in201 : in std_logic;
        in202 : in std_logic;
        in203 : in std_logic;
        in204 : in std_logic;
        in205 : in std_logic;
        in206 : in std_logic;
        in207 : in std_logic;
        in208 : in std_logic;
        in209 : in std_logic;
        in210 : in std_logic;
        in211 : in std_logic;
        in212 : in std_logic;
        in213 : in std_logic;
        in214 : in std_logic;
        in215 : in std_logic;
        in216 : in std_logic;
        in217 : in std_logic;
        in218 : in std_logic;
        in219 : in std_logic;
        in220 : in std_logic;
        in221 : in std_logic;
        in222 : in std_logic;
        in223 : in std_logic;
        in224 : in std_logic;
        in225 : in std_logic;
        in226 : in std_logic;
        in227 : in std_logic;
        in228 : in std_logic;
        in229 : in std_logic;
        in230 : in std_logic;
        in231 : in std_logic;
        in232 : in std_logic;
        in233 : in std_logic;
        in234 : in std_logic;
        in235 : in std_logic;
        in236 : in std_logic;
        in237 : in std_logic;
        in238 : in std_logic;
        in239 : in std_logic;
        in240 : in std_logic;
        in241 : in std_logic;
        in242 : in std_logic;
        in243 : in std_logic;
        in244 : in std_logic;
        in245 : in std_logic;
        in246 : in std_logic;
        in247 : in std_logic;
        in248 : in std_logic;
        in249 : in std_logic;
        in250 : in std_logic;
        in251 : in std_logic;
        in252 : in std_logic;
        in253 : in std_logic;
        in254 : in std_logic;
        in255 : in std_logic;
        in256 : in std_logic;
        in257 : in std_logic;
        in258 : in std_logic;
        in259 : in std_logic;
        in260 : in std_logic;
        in261 : in std_logic;
        in262 : in std_logic;
        in263 : in std_logic;
        in264 : in std_logic;
        in265 : in std_logic;
        in266 : in std_logic;
        in267 : in std_logic;
        in268 : in std_logic;
        in269 : in std_logic;
        in270 : in std_logic;
        in271 : in std_logic;
        in272 : in std_logic;
        in273 : in std_logic;
        in274 : in std_logic;
        in275 : in std_logic;
        in276 : in std_logic;
        in277 : in std_logic;
        in278 : in std_logic;
        in279 : in std_logic;
        in280 : in std_logic;
        in281 : in std_logic;
        in282 : in std_logic;
        in283 : in std_logic;
        in284 : in std_logic;
        in285 : in std_logic;
        in286 : in std_logic;
        in287 : in std_logic;
        in288 : in std_logic;
        in289 : in std_logic;
        in290 : in std_logic;
        in291 : in std_logic;
        in292 : in std_logic;
        in293 : in std_logic;
        in294 : in std_logic;
        in295 : in std_logic;
        in296 : in std_logic;
        in297 : in std_logic;
        in298 : in std_logic;
        in299 : in std_logic;
        in300 : in std_logic;
        in301 : in std_logic;
        in302 : in std_logic;
        in303 : in std_logic;
        in304 : in std_logic;
        in305 : in std_logic;
        in306 : in std_logic;
        in307 : in std_logic;
        in308 : in std_logic;
        in309 : in std_logic;
        in310 : in std_logic;
        in311 : in std_logic;
        in312 : in std_logic;
        in313 : in std_logic;
        in314 : in std_logic;
        in315 : in std_logic;
        in316 : in std_logic;
        in317 : in std_logic;
        in318 : in std_logic;
        in319 : in std_logic;
        in320 : in std_logic;
        in321 : in std_logic;
        in322 : in std_logic;
        in323 : in std_logic;
        in324 : in std_logic;
        in325 : in std_logic;
        in326 : in std_logic;
        in327 : in std_logic;
        in328 : in std_logic;
        in329 : in std_logic;
        in330 : in std_logic;
        in331 : in std_logic;
        in332 : in std_logic;
        in333 : in std_logic;
        in334 : in std_logic;
        in335 : in std_logic;
        in336 : in std_logic;
        in337 : in std_logic;
        in338 : in std_logic;
        in339 : in std_logic;
        in340 : in std_logic;
        in341 : in std_logic;
        in342 : in std_logic;
        in343 : in std_logic;
        in344 : in std_logic;
        in345 : in std_logic;
        in346 : in std_logic;
        in347 : in std_logic;
        in348 : in std_logic;
        in349 : in std_logic;
        in350 : in std_logic;
        in351 : in std_logic;
        in352 : in std_logic;
        in353 : in std_logic;
        in354 : in std_logic;
        in355 : in std_logic;
        in356 : in std_logic;
        in357 : in std_logic;
        in358 : in std_logic;
        in359 : in std_logic;
        in360 : in std_logic;
        in361 : in std_logic;
        in362 : in std_logic;
        in363 : in std_logic;
        in364 : in std_logic;
        in365 : in std_logic;
        in366 : in std_logic;
        in367 : in std_logic;
        in368 : in std_logic;
        in369 : in std_logic;
        in370 : in std_logic;
        in371 : in std_logic;
        in372 : in std_logic;
        in373 : in std_logic;
        in374 : in std_logic;
        in375 : in std_logic;
        in376 : in std_logic;
        in377 : in std_logic;
        in378 : in std_logic;
        in379 : in std_logic;
        in380 : in std_logic;
        in381 : in std_logic;
        in382 : in std_logic;
        in383 : in std_logic;
        in384 : in std_logic;
        in385 : in std_logic;
        in386 : in std_logic;
        in387 : in std_logic;
        in388 : in std_logic;
        in389 : in std_logic;
        in390 : in std_logic;
        in391 : in std_logic;
        in392 : in std_logic;
        in393 : in std_logic;
        in394 : in std_logic;
        in395 : in std_logic;
        in396 : in std_logic;
        in397 : in std_logic;
        in398 : in std_logic;
        in399 : in std_logic;
        in400 : in std_logic;
        in401 : in std_logic;
        in402 : in std_logic;
        in403 : in std_logic;
        in404 : in std_logic;
        in405 : in std_logic;
        in406 : in std_logic;
        in407 : in std_logic;
        in408 : in std_logic;
        in409 : in std_logic;
        in410 : in std_logic;
        in411 : in std_logic;
        in412 : in std_logic;
        in413 : in std_logic;
        in414 : in std_logic;
        in415 : in std_logic;
        in416 : in std_logic;
        in417 : in std_logic;
        in418 : in std_logic;
        in419 : in std_logic;
        in420 : in std_logic;
        in421 : in std_logic;
        in422 : in std_logic;
        in423 : in std_logic;
        in424 : in std_logic;
        in425 : in std_logic;
        in426 : in std_logic;
        in427 : in std_logic;
        in428 : in std_logic;
        in429 : in std_logic;
        in430 : in std_logic;
        in431 : in std_logic;
        in432 : in std_logic;
        in433 : in std_logic;
        in434 : in std_logic;
        in435 : in std_logic;
        in436 : in std_logic;
        in437 : in std_logic;
        in438 : in std_logic;
        in439 : in std_logic;
        in440 : in std_logic;
        in441 : in std_logic;
        in442 : in std_logic;
        in443 : in std_logic;
        in444 : in std_logic;
        in445 : in std_logic;
        in446 : in std_logic;
        in447 : in std_logic;
        in448 : in std_logic;
        in449 : in std_logic;
        in450 : in std_logic;
        in451 : in std_logic;
        in452 : in std_logic;
        in453 : in std_logic;
        in454 : in std_logic;
        in455 : in std_logic;
        in456 : in std_logic;
        in457 : in std_logic;
        in458 : in std_logic;
        in459 : in std_logic;
        in460 : in std_logic;
        in461 : in std_logic;
        in462 : in std_logic;
        in463 : in std_logic;
        in464 : in std_logic;
        in465 : in std_logic;
        in466 : in std_logic;
        in467 : in std_logic;
        in468 : in std_logic;
        in469 : in std_logic;
        in470 : in std_logic;
        in471 : in std_logic;
        in472 : in std_logic;
        in473 : in std_logic;
        in474 : in std_logic;
        in475 : in std_logic;
        in476 : in std_logic;
        in477 : in std_logic;
        in478 : in std_logic;
        in479 : in std_logic;
        in480 : in std_logic;
        in481 : in std_logic;
        in482 : in std_logic;
        in483 : in std_logic;
        in484 : in std_logic;
        in485 : in std_logic;
        in486 : in std_logic;
        in487 : in std_logic;
        in488 : in std_logic;
        in489 : in std_logic;
        in490 : in std_logic;
        in491 : in std_logic;
        in492 : in std_logic;
        in493 : in std_logic;
        in494 : in std_logic;
        in495 : in std_logic;
        in496 : in std_logic;
        in497 : in std_logic;
        in498 : in std_logic;
        in499 : in std_logic;
        in500 : in std_logic;
        in501 : in std_logic;
        in502 : in std_logic;
        in503 : in std_logic;
        in504 : in std_logic;
        in505 : in std_logic;
        in506 : in std_logic;
        in507 : in std_logic;
        in508 : in std_logic;
        in509 : in std_logic;
        in510 : in std_logic;
        in511 : in std_logic;
        mux_out : out std_logic
    );
end entity mux_512to1;

architecture behavior of mux_512to1 is


begin

    selection : process(mux_sel, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19, in20, in21, in22, in23, in24, in25, in26, in27, in28, in29, in30, in31, in32, in33, in34, in35, in36, in37, in38, in39, in40, in41, in42, in43, in44, in45, in46, in47, in48, in49, in50, in51, in52, in53, in54, in55, in56, in57, in58, in59, in60, in61, in62, in63, in64, in65, in66, in67, in68, in69, in70, in71, in72, in73, in74, in75, in76, in77, in78, in79, in80, in81, in82, in83, in84, in85, in86, in87, in88, in89, in90, in91, in92, in93, in94, in95, in96, in97, in98, in99, in100, in101, in102, in103, in104, in105, in106, in107, in108, in109, in110, in111, in112, in113, in114, in115, in116, in117, in118, in119, in120, in121, in122, in123, in124, in125, in126, in127, in128, in129, in130, in131, in132, in133, in134, in135, in136, in137, in138, in139, in140, in141, in142, in143, in144, in145, in146, in147, in148, in149, in150, in151, in152, in153, in154, in155, in156, in157, in158, in159, in160, in161, in162, in163, in164, in165, in166, in167, in168, in169, in170, in171, in172, in173, in174, in175, in176, in177, in178, in179, in180, in181, in182, in183, in184, in185, in186, in187, in188, in189, in190, in191, in192, in193, in194, in195, in196, in197, in198, in199, in200, in201, in202, in203, in204, in205, in206, in207, in208, in209, in210, in211, in212, in213, in214, in215, in216, in217, in218, in219, in220, in221, in222, in223, in224, in225, in226, in227, in228, in229, in230, in231, in232, in233, in234, in235, in236, in237, in238, in239, in240, in241, in242, in243, in244, in245, in246, in247, in248, in249, in250, in251, in252, in253, in254, in255, in256, in257, in258, in259, in260, in261, in262, in263, in264, in265, in266, in267, in268, in269, in270, in271, in272, in273, in274, in275, in276, in277, in278, in279, in280, in281, in282, in283, in284, in285, in286, in287, in288, in289, in290, in291, in292, in293, in294, in295, in296, in297, in298, in299, in300, in301, in302, in303, in304, in305, in306, in307, in308, in309, in310, in311, in312, in313, in314, in315, in316, in317, in318, in319, in320, in321, in322, in323, in324, in325, in326, in327, in328, in329, in330, in331, in332, in333, in334, in335, in336, in337, in338, in339, in340, in341, in342, in343, in344, in345, in346, in347, in348, in349, in350, in351, in352, in353, in354, in355, in356, in357, in358, in359, in360, in361, in362, in363, in364, in365, in366, in367, in368, in369, in370, in371, in372, in373, in374, in375, in376, in377, in378, in379, in380, in381, in382, in383, in384, in385, in386, in387, in388, in389, in390, in391, in392, in393, in394, in395, in396, in397, in398, in399, in400, in401, in402, in403, in404, in405, in406, in407, in408, in409, in410, in411, in412, in413, in414, in415, in416, in417, in418, in419, in420, in421, in422, in423, in424, in425, in426, in427, in428, in429, in430, in431, in432, in433, in434, in435, in436, in437, in438, in439, in440, in441, in442, in443, in444, in445, in446, in447, in448, in449, in450, in451, in452, in453, in454, in455, in456, in457, in458, in459, in460, in461, in462, in463, in464, in465, in466, in467, in468, in469, in470, in471, in472, in473, in474, in475, in476, in477, in478, in479, in480, in481, in482, in483, in484, in485, in486, in487, in488, in489, in490, in491, in492, in493, in494, in495, in496, in497, in498, in499, in500, in501, in502, in503, in504, in505, in506, in507, in508, in509, in510, in511 )
    begin

        case mux_sel is

            when "000000000" =>
                mux_out <= in0;


            when "000000001" =>
                mux_out <= in1;


            when "000000010" =>
                mux_out <= in2;


            when "000000011" =>
                mux_out <= in3;


            when "000000100" =>
                mux_out <= in4;


            when "000000101" =>
                mux_out <= in5;


            when "000000110" =>
                mux_out <= in6;


            when "000000111" =>
                mux_out <= in7;


            when "000001000" =>
                mux_out <= in8;


            when "000001001" =>
                mux_out <= in9;


            when "000001010" =>
                mux_out <= in10;


            when "000001011" =>
                mux_out <= in11;


            when "000001100" =>
                mux_out <= in12;


            when "000001101" =>
                mux_out <= in13;


            when "000001110" =>
                mux_out <= in14;


            when "000001111" =>
                mux_out <= in15;


            when "000010000" =>
                mux_out <= in16;


            when "000010001" =>
                mux_out <= in17;


            when "000010010" =>
                mux_out <= in18;


            when "000010011" =>
                mux_out <= in19;


            when "000010100" =>
                mux_out <= in20;


            when "000010101" =>
                mux_out <= in21;


            when "000010110" =>
                mux_out <= in22;


            when "000010111" =>
                mux_out <= in23;


            when "000011000" =>
                mux_out <= in24;


            when "000011001" =>
                mux_out <= in25;


            when "000011010" =>
                mux_out <= in26;


            when "000011011" =>
                mux_out <= in27;


            when "000011100" =>
                mux_out <= in28;


            when "000011101" =>
                mux_out <= in29;


            when "000011110" =>
                mux_out <= in30;


            when "000011111" =>
                mux_out <= in31;


            when "000100000" =>
                mux_out <= in32;


            when "000100001" =>
                mux_out <= in33;


            when "000100010" =>
                mux_out <= in34;


            when "000100011" =>
                mux_out <= in35;


            when "000100100" =>
                mux_out <= in36;


            when "000100101" =>
                mux_out <= in37;


            when "000100110" =>
                mux_out <= in38;


            when "000100111" =>
                mux_out <= in39;


            when "000101000" =>
                mux_out <= in40;


            when "000101001" =>
                mux_out <= in41;


            when "000101010" =>
                mux_out <= in42;


            when "000101011" =>
                mux_out <= in43;


            when "000101100" =>
                mux_out <= in44;


            when "000101101" =>
                mux_out <= in45;


            when "000101110" =>
                mux_out <= in46;


            when "000101111" =>
                mux_out <= in47;


            when "000110000" =>
                mux_out <= in48;


            when "000110001" =>
                mux_out <= in49;


            when "000110010" =>
                mux_out <= in50;


            when "000110011" =>
                mux_out <= in51;


            when "000110100" =>
                mux_out <= in52;


            when "000110101" =>
                mux_out <= in53;


            when "000110110" =>
                mux_out <= in54;


            when "000110111" =>
                mux_out <= in55;


            when "000111000" =>
                mux_out <= in56;


            when "000111001" =>
                mux_out <= in57;


            when "000111010" =>
                mux_out <= in58;


            when "000111011" =>
                mux_out <= in59;


            when "000111100" =>
                mux_out <= in60;


            when "000111101" =>
                mux_out <= in61;


            when "000111110" =>
                mux_out <= in62;


            when "000111111" =>
                mux_out <= in63;


            when "001000000" =>
                mux_out <= in64;


            when "001000001" =>
                mux_out <= in65;


            when "001000010" =>
                mux_out <= in66;


            when "001000011" =>
                mux_out <= in67;


            when "001000100" =>
                mux_out <= in68;


            when "001000101" =>
                mux_out <= in69;


            when "001000110" =>
                mux_out <= in70;


            when "001000111" =>
                mux_out <= in71;


            when "001001000" =>
                mux_out <= in72;


            when "001001001" =>
                mux_out <= in73;


            when "001001010" =>
                mux_out <= in74;


            when "001001011" =>
                mux_out <= in75;


            when "001001100" =>
                mux_out <= in76;


            when "001001101" =>
                mux_out <= in77;


            when "001001110" =>
                mux_out <= in78;


            when "001001111" =>
                mux_out <= in79;


            when "001010000" =>
                mux_out <= in80;


            when "001010001" =>
                mux_out <= in81;


            when "001010010" =>
                mux_out <= in82;


            when "001010011" =>
                mux_out <= in83;


            when "001010100" =>
                mux_out <= in84;


            when "001010101" =>
                mux_out <= in85;


            when "001010110" =>
                mux_out <= in86;


            when "001010111" =>
                mux_out <= in87;


            when "001011000" =>
                mux_out <= in88;


            when "001011001" =>
                mux_out <= in89;


            when "001011010" =>
                mux_out <= in90;


            when "001011011" =>
                mux_out <= in91;


            when "001011100" =>
                mux_out <= in92;


            when "001011101" =>
                mux_out <= in93;


            when "001011110" =>
                mux_out <= in94;


            when "001011111" =>
                mux_out <= in95;


            when "001100000" =>
                mux_out <= in96;


            when "001100001" =>
                mux_out <= in97;


            when "001100010" =>
                mux_out <= in98;


            when "001100011" =>
                mux_out <= in99;


            when "001100100" =>
                mux_out <= in100;


            when "001100101" =>
                mux_out <= in101;


            when "001100110" =>
                mux_out <= in102;


            when "001100111" =>
                mux_out <= in103;


            when "001101000" =>
                mux_out <= in104;


            when "001101001" =>
                mux_out <= in105;


            when "001101010" =>
                mux_out <= in106;


            when "001101011" =>
                mux_out <= in107;


            when "001101100" =>
                mux_out <= in108;


            when "001101101" =>
                mux_out <= in109;


            when "001101110" =>
                mux_out <= in110;


            when "001101111" =>
                mux_out <= in111;


            when "001110000" =>
                mux_out <= in112;


            when "001110001" =>
                mux_out <= in113;


            when "001110010" =>
                mux_out <= in114;


            when "001110011" =>
                mux_out <= in115;


            when "001110100" =>
                mux_out <= in116;


            when "001110101" =>
                mux_out <= in117;


            when "001110110" =>
                mux_out <= in118;


            when "001110111" =>
                mux_out <= in119;


            when "001111000" =>
                mux_out <= in120;


            when "001111001" =>
                mux_out <= in121;


            when "001111010" =>
                mux_out <= in122;


            when "001111011" =>
                mux_out <= in123;


            when "001111100" =>
                mux_out <= in124;


            when "001111101" =>
                mux_out <= in125;


            when "001111110" =>
                mux_out <= in126;


            when "001111111" =>
                mux_out <= in127;


            when "010000000" =>
                mux_out <= in128;


            when "010000001" =>
                mux_out <= in129;


            when "010000010" =>
                mux_out <= in130;


            when "010000011" =>
                mux_out <= in131;


            when "010000100" =>
                mux_out <= in132;


            when "010000101" =>
                mux_out <= in133;


            when "010000110" =>
                mux_out <= in134;


            when "010000111" =>
                mux_out <= in135;


            when "010001000" =>
                mux_out <= in136;


            when "010001001" =>
                mux_out <= in137;


            when "010001010" =>
                mux_out <= in138;


            when "010001011" =>
                mux_out <= in139;


            when "010001100" =>
                mux_out <= in140;


            when "010001101" =>
                mux_out <= in141;


            when "010001110" =>
                mux_out <= in142;


            when "010001111" =>
                mux_out <= in143;


            when "010010000" =>
                mux_out <= in144;


            when "010010001" =>
                mux_out <= in145;


            when "010010010" =>
                mux_out <= in146;


            when "010010011" =>
                mux_out <= in147;


            when "010010100" =>
                mux_out <= in148;


            when "010010101" =>
                mux_out <= in149;


            when "010010110" =>
                mux_out <= in150;


            when "010010111" =>
                mux_out <= in151;


            when "010011000" =>
                mux_out <= in152;


            when "010011001" =>
                mux_out <= in153;


            when "010011010" =>
                mux_out <= in154;


            when "010011011" =>
                mux_out <= in155;


            when "010011100" =>
                mux_out <= in156;


            when "010011101" =>
                mux_out <= in157;


            when "010011110" =>
                mux_out <= in158;


            when "010011111" =>
                mux_out <= in159;


            when "010100000" =>
                mux_out <= in160;


            when "010100001" =>
                mux_out <= in161;


            when "010100010" =>
                mux_out <= in162;


            when "010100011" =>
                mux_out <= in163;


            when "010100100" =>
                mux_out <= in164;


            when "010100101" =>
                mux_out <= in165;


            when "010100110" =>
                mux_out <= in166;


            when "010100111" =>
                mux_out <= in167;


            when "010101000" =>
                mux_out <= in168;


            when "010101001" =>
                mux_out <= in169;


            when "010101010" =>
                mux_out <= in170;


            when "010101011" =>
                mux_out <= in171;


            when "010101100" =>
                mux_out <= in172;


            when "010101101" =>
                mux_out <= in173;


            when "010101110" =>
                mux_out <= in174;


            when "010101111" =>
                mux_out <= in175;


            when "010110000" =>
                mux_out <= in176;


            when "010110001" =>
                mux_out <= in177;


            when "010110010" =>
                mux_out <= in178;


            when "010110011" =>
                mux_out <= in179;


            when "010110100" =>
                mux_out <= in180;


            when "010110101" =>
                mux_out <= in181;


            when "010110110" =>
                mux_out <= in182;


            when "010110111" =>
                mux_out <= in183;


            when "010111000" =>
                mux_out <= in184;


            when "010111001" =>
                mux_out <= in185;


            when "010111010" =>
                mux_out <= in186;


            when "010111011" =>
                mux_out <= in187;


            when "010111100" =>
                mux_out <= in188;


            when "010111101" =>
                mux_out <= in189;


            when "010111110" =>
                mux_out <= in190;


            when "010111111" =>
                mux_out <= in191;


            when "011000000" =>
                mux_out <= in192;


            when "011000001" =>
                mux_out <= in193;


            when "011000010" =>
                mux_out <= in194;


            when "011000011" =>
                mux_out <= in195;


            when "011000100" =>
                mux_out <= in196;


            when "011000101" =>
                mux_out <= in197;


            when "011000110" =>
                mux_out <= in198;


            when "011000111" =>
                mux_out <= in199;


            when "011001000" =>
                mux_out <= in200;


            when "011001001" =>
                mux_out <= in201;


            when "011001010" =>
                mux_out <= in202;


            when "011001011" =>
                mux_out <= in203;


            when "011001100" =>
                mux_out <= in204;


            when "011001101" =>
                mux_out <= in205;


            when "011001110" =>
                mux_out <= in206;


            when "011001111" =>
                mux_out <= in207;


            when "011010000" =>
                mux_out <= in208;


            when "011010001" =>
                mux_out <= in209;


            when "011010010" =>
                mux_out <= in210;


            when "011010011" =>
                mux_out <= in211;


            when "011010100" =>
                mux_out <= in212;


            when "011010101" =>
                mux_out <= in213;


            when "011010110" =>
                mux_out <= in214;


            when "011010111" =>
                mux_out <= in215;


            when "011011000" =>
                mux_out <= in216;


            when "011011001" =>
                mux_out <= in217;


            when "011011010" =>
                mux_out <= in218;


            when "011011011" =>
                mux_out <= in219;


            when "011011100" =>
                mux_out <= in220;


            when "011011101" =>
                mux_out <= in221;


            when "011011110" =>
                mux_out <= in222;


            when "011011111" =>
                mux_out <= in223;


            when "011100000" =>
                mux_out <= in224;


            when "011100001" =>
                mux_out <= in225;


            when "011100010" =>
                mux_out <= in226;


            when "011100011" =>
                mux_out <= in227;


            when "011100100" =>
                mux_out <= in228;


            when "011100101" =>
                mux_out <= in229;


            when "011100110" =>
                mux_out <= in230;


            when "011100111" =>
                mux_out <= in231;


            when "011101000" =>
                mux_out <= in232;


            when "011101001" =>
                mux_out <= in233;


            when "011101010" =>
                mux_out <= in234;


            when "011101011" =>
                mux_out <= in235;


            when "011101100" =>
                mux_out <= in236;


            when "011101101" =>
                mux_out <= in237;


            when "011101110" =>
                mux_out <= in238;


            when "011101111" =>
                mux_out <= in239;


            when "011110000" =>
                mux_out <= in240;


            when "011110001" =>
                mux_out <= in241;


            when "011110010" =>
                mux_out <= in242;


            when "011110011" =>
                mux_out <= in243;


            when "011110100" =>
                mux_out <= in244;


            when "011110101" =>
                mux_out <= in245;


            when "011110110" =>
                mux_out <= in246;


            when "011110111" =>
                mux_out <= in247;


            when "011111000" =>
                mux_out <= in248;


            when "011111001" =>
                mux_out <= in249;


            when "011111010" =>
                mux_out <= in250;


            when "011111011" =>
                mux_out <= in251;


            when "011111100" =>
                mux_out <= in252;


            when "011111101" =>
                mux_out <= in253;


            when "011111110" =>
                mux_out <= in254;


            when "011111111" =>
                mux_out <= in255;


            when "100000000" =>
                mux_out <= in256;


            when "100000001" =>
                mux_out <= in257;


            when "100000010" =>
                mux_out <= in258;


            when "100000011" =>
                mux_out <= in259;


            when "100000100" =>
                mux_out <= in260;


            when "100000101" =>
                mux_out <= in261;


            when "100000110" =>
                mux_out <= in262;


            when "100000111" =>
                mux_out <= in263;


            when "100001000" =>
                mux_out <= in264;


            when "100001001" =>
                mux_out <= in265;


            when "100001010" =>
                mux_out <= in266;


            when "100001011" =>
                mux_out <= in267;


            when "100001100" =>
                mux_out <= in268;


            when "100001101" =>
                mux_out <= in269;


            when "100001110" =>
                mux_out <= in270;


            when "100001111" =>
                mux_out <= in271;


            when "100010000" =>
                mux_out <= in272;


            when "100010001" =>
                mux_out <= in273;


            when "100010010" =>
                mux_out <= in274;


            when "100010011" =>
                mux_out <= in275;


            when "100010100" =>
                mux_out <= in276;


            when "100010101" =>
                mux_out <= in277;


            when "100010110" =>
                mux_out <= in278;


            when "100010111" =>
                mux_out <= in279;


            when "100011000" =>
                mux_out <= in280;


            when "100011001" =>
                mux_out <= in281;


            when "100011010" =>
                mux_out <= in282;


            when "100011011" =>
                mux_out <= in283;


            when "100011100" =>
                mux_out <= in284;


            when "100011101" =>
                mux_out <= in285;


            when "100011110" =>
                mux_out <= in286;


            when "100011111" =>
                mux_out <= in287;


            when "100100000" =>
                mux_out <= in288;


            when "100100001" =>
                mux_out <= in289;


            when "100100010" =>
                mux_out <= in290;


            when "100100011" =>
                mux_out <= in291;


            when "100100100" =>
                mux_out <= in292;


            when "100100101" =>
                mux_out <= in293;


            when "100100110" =>
                mux_out <= in294;


            when "100100111" =>
                mux_out <= in295;


            when "100101000" =>
                mux_out <= in296;


            when "100101001" =>
                mux_out <= in297;


            when "100101010" =>
                mux_out <= in298;


            when "100101011" =>
                mux_out <= in299;


            when "100101100" =>
                mux_out <= in300;


            when "100101101" =>
                mux_out <= in301;


            when "100101110" =>
                mux_out <= in302;


            when "100101111" =>
                mux_out <= in303;


            when "100110000" =>
                mux_out <= in304;


            when "100110001" =>
                mux_out <= in305;


            when "100110010" =>
                mux_out <= in306;


            when "100110011" =>
                mux_out <= in307;


            when "100110100" =>
                mux_out <= in308;


            when "100110101" =>
                mux_out <= in309;


            when "100110110" =>
                mux_out <= in310;


            when "100110111" =>
                mux_out <= in311;


            when "100111000" =>
                mux_out <= in312;


            when "100111001" =>
                mux_out <= in313;


            when "100111010" =>
                mux_out <= in314;


            when "100111011" =>
                mux_out <= in315;


            when "100111100" =>
                mux_out <= in316;


            when "100111101" =>
                mux_out <= in317;


            when "100111110" =>
                mux_out <= in318;


            when "100111111" =>
                mux_out <= in319;


            when "101000000" =>
                mux_out <= in320;


            when "101000001" =>
                mux_out <= in321;


            when "101000010" =>
                mux_out <= in322;


            when "101000011" =>
                mux_out <= in323;


            when "101000100" =>
                mux_out <= in324;


            when "101000101" =>
                mux_out <= in325;


            when "101000110" =>
                mux_out <= in326;


            when "101000111" =>
                mux_out <= in327;


            when "101001000" =>
                mux_out <= in328;


            when "101001001" =>
                mux_out <= in329;


            when "101001010" =>
                mux_out <= in330;


            when "101001011" =>
                mux_out <= in331;


            when "101001100" =>
                mux_out <= in332;


            when "101001101" =>
                mux_out <= in333;


            when "101001110" =>
                mux_out <= in334;


            when "101001111" =>
                mux_out <= in335;


            when "101010000" =>
                mux_out <= in336;


            when "101010001" =>
                mux_out <= in337;


            when "101010010" =>
                mux_out <= in338;


            when "101010011" =>
                mux_out <= in339;


            when "101010100" =>
                mux_out <= in340;


            when "101010101" =>
                mux_out <= in341;


            when "101010110" =>
                mux_out <= in342;


            when "101010111" =>
                mux_out <= in343;


            when "101011000" =>
                mux_out <= in344;


            when "101011001" =>
                mux_out <= in345;


            when "101011010" =>
                mux_out <= in346;


            when "101011011" =>
                mux_out <= in347;


            when "101011100" =>
                mux_out <= in348;


            when "101011101" =>
                mux_out <= in349;


            when "101011110" =>
                mux_out <= in350;


            when "101011111" =>
                mux_out <= in351;


            when "101100000" =>
                mux_out <= in352;


            when "101100001" =>
                mux_out <= in353;


            when "101100010" =>
                mux_out <= in354;


            when "101100011" =>
                mux_out <= in355;


            when "101100100" =>
                mux_out <= in356;


            when "101100101" =>
                mux_out <= in357;


            when "101100110" =>
                mux_out <= in358;


            when "101100111" =>
                mux_out <= in359;


            when "101101000" =>
                mux_out <= in360;


            when "101101001" =>
                mux_out <= in361;


            when "101101010" =>
                mux_out <= in362;


            when "101101011" =>
                mux_out <= in363;


            when "101101100" =>
                mux_out <= in364;


            when "101101101" =>
                mux_out <= in365;


            when "101101110" =>
                mux_out <= in366;


            when "101101111" =>
                mux_out <= in367;


            when "101110000" =>
                mux_out <= in368;


            when "101110001" =>
                mux_out <= in369;


            when "101110010" =>
                mux_out <= in370;


            when "101110011" =>
                mux_out <= in371;


            when "101110100" =>
                mux_out <= in372;


            when "101110101" =>
                mux_out <= in373;


            when "101110110" =>
                mux_out <= in374;


            when "101110111" =>
                mux_out <= in375;


            when "101111000" =>
                mux_out <= in376;


            when "101111001" =>
                mux_out <= in377;


            when "101111010" =>
                mux_out <= in378;


            when "101111011" =>
                mux_out <= in379;


            when "101111100" =>
                mux_out <= in380;


            when "101111101" =>
                mux_out <= in381;


            when "101111110" =>
                mux_out <= in382;


            when "101111111" =>
                mux_out <= in383;


            when "110000000" =>
                mux_out <= in384;


            when "110000001" =>
                mux_out <= in385;


            when "110000010" =>
                mux_out <= in386;


            when "110000011" =>
                mux_out <= in387;


            when "110000100" =>
                mux_out <= in388;


            when "110000101" =>
                mux_out <= in389;


            when "110000110" =>
                mux_out <= in390;


            when "110000111" =>
                mux_out <= in391;


            when "110001000" =>
                mux_out <= in392;


            when "110001001" =>
                mux_out <= in393;


            when "110001010" =>
                mux_out <= in394;


            when "110001011" =>
                mux_out <= in395;


            when "110001100" =>
                mux_out <= in396;


            when "110001101" =>
                mux_out <= in397;


            when "110001110" =>
                mux_out <= in398;


            when "110001111" =>
                mux_out <= in399;


            when "110010000" =>
                mux_out <= in400;


            when "110010001" =>
                mux_out <= in401;


            when "110010010" =>
                mux_out <= in402;


            when "110010011" =>
                mux_out <= in403;


            when "110010100" =>
                mux_out <= in404;


            when "110010101" =>
                mux_out <= in405;


            when "110010110" =>
                mux_out <= in406;


            when "110010111" =>
                mux_out <= in407;


            when "110011000" =>
                mux_out <= in408;


            when "110011001" =>
                mux_out <= in409;


            when "110011010" =>
                mux_out <= in410;


            when "110011011" =>
                mux_out <= in411;


            when "110011100" =>
                mux_out <= in412;


            when "110011101" =>
                mux_out <= in413;


            when "110011110" =>
                mux_out <= in414;


            when "110011111" =>
                mux_out <= in415;


            when "110100000" =>
                mux_out <= in416;


            when "110100001" =>
                mux_out <= in417;


            when "110100010" =>
                mux_out <= in418;


            when "110100011" =>
                mux_out <= in419;


            when "110100100" =>
                mux_out <= in420;


            when "110100101" =>
                mux_out <= in421;


            when "110100110" =>
                mux_out <= in422;


            when "110100111" =>
                mux_out <= in423;


            when "110101000" =>
                mux_out <= in424;


            when "110101001" =>
                mux_out <= in425;


            when "110101010" =>
                mux_out <= in426;


            when "110101011" =>
                mux_out <= in427;


            when "110101100" =>
                mux_out <= in428;


            when "110101101" =>
                mux_out <= in429;


            when "110101110" =>
                mux_out <= in430;


            when "110101111" =>
                mux_out <= in431;


            when "110110000" =>
                mux_out <= in432;


            when "110110001" =>
                mux_out <= in433;


            when "110110010" =>
                mux_out <= in434;


            when "110110011" =>
                mux_out <= in435;


            when "110110100" =>
                mux_out <= in436;


            when "110110101" =>
                mux_out <= in437;


            when "110110110" =>
                mux_out <= in438;


            when "110110111" =>
                mux_out <= in439;


            when "110111000" =>
                mux_out <= in440;


            when "110111001" =>
                mux_out <= in441;


            when "110111010" =>
                mux_out <= in442;


            when "110111011" =>
                mux_out <= in443;


            when "110111100" =>
                mux_out <= in444;


            when "110111101" =>
                mux_out <= in445;


            when "110111110" =>
                mux_out <= in446;


            when "110111111" =>
                mux_out <= in447;


            when "111000000" =>
                mux_out <= in448;


            when "111000001" =>
                mux_out <= in449;


            when "111000010" =>
                mux_out <= in450;


            when "111000011" =>
                mux_out <= in451;


            when "111000100" =>
                mux_out <= in452;


            when "111000101" =>
                mux_out <= in453;


            when "111000110" =>
                mux_out <= in454;


            when "111000111" =>
                mux_out <= in455;


            when "111001000" =>
                mux_out <= in456;


            when "111001001" =>
                mux_out <= in457;


            when "111001010" =>
                mux_out <= in458;


            when "111001011" =>
                mux_out <= in459;


            when "111001100" =>
                mux_out <= in460;


            when "111001101" =>
                mux_out <= in461;


            when "111001110" =>
                mux_out <= in462;


            when "111001111" =>
                mux_out <= in463;


            when "111010000" =>
                mux_out <= in464;


            when "111010001" =>
                mux_out <= in465;


            when "111010010" =>
                mux_out <= in466;


            when "111010011" =>
                mux_out <= in467;


            when "111010100" =>
                mux_out <= in468;


            when "111010101" =>
                mux_out <= in469;


            when "111010110" =>
                mux_out <= in470;


            when "111010111" =>
                mux_out <= in471;


            when "111011000" =>
                mux_out <= in472;


            when "111011001" =>
                mux_out <= in473;


            when "111011010" =>
                mux_out <= in474;


            when "111011011" =>
                mux_out <= in475;


            when "111011100" =>
                mux_out <= in476;


            when "111011101" =>
                mux_out <= in477;


            when "111011110" =>
                mux_out <= in478;


            when "111011111" =>
                mux_out <= in479;


            when "111100000" =>
                mux_out <= in480;


            when "111100001" =>
                mux_out <= in481;


            when "111100010" =>
                mux_out <= in482;


            when "111100011" =>
                mux_out <= in483;


            when "111100100" =>
                mux_out <= in484;


            when "111100101" =>
                mux_out <= in485;


            when "111100110" =>
                mux_out <= in486;


            when "111100111" =>
                mux_out <= in487;


            when "111101000" =>
                mux_out <= in488;


            when "111101001" =>
                mux_out <= in489;


            when "111101010" =>
                mux_out <= in490;


            when "111101011" =>
                mux_out <= in491;


            when "111101100" =>
                mux_out <= in492;


            when "111101101" =>
                mux_out <= in493;


            when "111101110" =>
                mux_out <= in494;


            when "111101111" =>
                mux_out <= in495;


            when "111110000" =>
                mux_out <= in496;


            when "111110001" =>
                mux_out <= in497;


            when "111110010" =>
                mux_out <= in498;


            when "111110011" =>
                mux_out <= in499;


            when "111110100" =>
                mux_out <= in500;


            when "111110101" =>
                mux_out <= in501;


            when "111110110" =>
                mux_out <= in502;


            when "111110111" =>
                mux_out <= in503;


            when "111111000" =>
                mux_out <= in504;


            when "111111001" =>
                mux_out <= in505;


            when "111111010" =>
                mux_out <= in506;


            when "111111011" =>
                mux_out <= in507;


            when "111111100" =>
                mux_out <= in508;


            when "111111101" =>
                mux_out <= in509;


            when "111111110" =>
                mux_out <= in510;


            when others =>
                mux_out <= in511;


        end case;

    end process selection;


end architecture behavior;

