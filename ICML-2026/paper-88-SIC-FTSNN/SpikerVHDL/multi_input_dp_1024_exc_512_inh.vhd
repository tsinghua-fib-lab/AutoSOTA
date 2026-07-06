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


entity multi_input_dp_1024_exc_512_inh is
    generic (
        n_exc_inputs : integer := 1024;
        n_inh_inputs : integer := 512;
        exc_cnt_bitwidth : integer := 10;
        inh_cnt_bitwidth : integer := 9
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
end entity multi_input_dp_1024_exc_512_inh;

architecture behavior of multi_input_dp_1024_exc_512_inh is


    component generic_or is
        generic (
            N : integer := 1024
        );
        port (
            or_in : in std_logic_vector(N-1 downto 0);
            or_out : out std_logic
        );
    end component;

    component reg_sync_rst is
        generic (
            N : integer := 1024
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
            N : integer := 10
        );
        port (
            in0 : in std_logic_vector(N-1 downto 0);
            in1 : in std_logic_vector(N-1 downto 0);
            cmp_out : out std_logic
        );
    end component;

    component mux_1024to1 is
        port (
            mux_sel : in std_logic_vector(9 downto 0);
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
            in512 : in std_logic;
            in513 : in std_logic;
            in514 : in std_logic;
            in515 : in std_logic;
            in516 : in std_logic;
            in517 : in std_logic;
            in518 : in std_logic;
            in519 : in std_logic;
            in520 : in std_logic;
            in521 : in std_logic;
            in522 : in std_logic;
            in523 : in std_logic;
            in524 : in std_logic;
            in525 : in std_logic;
            in526 : in std_logic;
            in527 : in std_logic;
            in528 : in std_logic;
            in529 : in std_logic;
            in530 : in std_logic;
            in531 : in std_logic;
            in532 : in std_logic;
            in533 : in std_logic;
            in534 : in std_logic;
            in535 : in std_logic;
            in536 : in std_logic;
            in537 : in std_logic;
            in538 : in std_logic;
            in539 : in std_logic;
            in540 : in std_logic;
            in541 : in std_logic;
            in542 : in std_logic;
            in543 : in std_logic;
            in544 : in std_logic;
            in545 : in std_logic;
            in546 : in std_logic;
            in547 : in std_logic;
            in548 : in std_logic;
            in549 : in std_logic;
            in550 : in std_logic;
            in551 : in std_logic;
            in552 : in std_logic;
            in553 : in std_logic;
            in554 : in std_logic;
            in555 : in std_logic;
            in556 : in std_logic;
            in557 : in std_logic;
            in558 : in std_logic;
            in559 : in std_logic;
            in560 : in std_logic;
            in561 : in std_logic;
            in562 : in std_logic;
            in563 : in std_logic;
            in564 : in std_logic;
            in565 : in std_logic;
            in566 : in std_logic;
            in567 : in std_logic;
            in568 : in std_logic;
            in569 : in std_logic;
            in570 : in std_logic;
            in571 : in std_logic;
            in572 : in std_logic;
            in573 : in std_logic;
            in574 : in std_logic;
            in575 : in std_logic;
            in576 : in std_logic;
            in577 : in std_logic;
            in578 : in std_logic;
            in579 : in std_logic;
            in580 : in std_logic;
            in581 : in std_logic;
            in582 : in std_logic;
            in583 : in std_logic;
            in584 : in std_logic;
            in585 : in std_logic;
            in586 : in std_logic;
            in587 : in std_logic;
            in588 : in std_logic;
            in589 : in std_logic;
            in590 : in std_logic;
            in591 : in std_logic;
            in592 : in std_logic;
            in593 : in std_logic;
            in594 : in std_logic;
            in595 : in std_logic;
            in596 : in std_logic;
            in597 : in std_logic;
            in598 : in std_logic;
            in599 : in std_logic;
            in600 : in std_logic;
            in601 : in std_logic;
            in602 : in std_logic;
            in603 : in std_logic;
            in604 : in std_logic;
            in605 : in std_logic;
            in606 : in std_logic;
            in607 : in std_logic;
            in608 : in std_logic;
            in609 : in std_logic;
            in610 : in std_logic;
            in611 : in std_logic;
            in612 : in std_logic;
            in613 : in std_logic;
            in614 : in std_logic;
            in615 : in std_logic;
            in616 : in std_logic;
            in617 : in std_logic;
            in618 : in std_logic;
            in619 : in std_logic;
            in620 : in std_logic;
            in621 : in std_logic;
            in622 : in std_logic;
            in623 : in std_logic;
            in624 : in std_logic;
            in625 : in std_logic;
            in626 : in std_logic;
            in627 : in std_logic;
            in628 : in std_logic;
            in629 : in std_logic;
            in630 : in std_logic;
            in631 : in std_logic;
            in632 : in std_logic;
            in633 : in std_logic;
            in634 : in std_logic;
            in635 : in std_logic;
            in636 : in std_logic;
            in637 : in std_logic;
            in638 : in std_logic;
            in639 : in std_logic;
            in640 : in std_logic;
            in641 : in std_logic;
            in642 : in std_logic;
            in643 : in std_logic;
            in644 : in std_logic;
            in645 : in std_logic;
            in646 : in std_logic;
            in647 : in std_logic;
            in648 : in std_logic;
            in649 : in std_logic;
            in650 : in std_logic;
            in651 : in std_logic;
            in652 : in std_logic;
            in653 : in std_logic;
            in654 : in std_logic;
            in655 : in std_logic;
            in656 : in std_logic;
            in657 : in std_logic;
            in658 : in std_logic;
            in659 : in std_logic;
            in660 : in std_logic;
            in661 : in std_logic;
            in662 : in std_logic;
            in663 : in std_logic;
            in664 : in std_logic;
            in665 : in std_logic;
            in666 : in std_logic;
            in667 : in std_logic;
            in668 : in std_logic;
            in669 : in std_logic;
            in670 : in std_logic;
            in671 : in std_logic;
            in672 : in std_logic;
            in673 : in std_logic;
            in674 : in std_logic;
            in675 : in std_logic;
            in676 : in std_logic;
            in677 : in std_logic;
            in678 : in std_logic;
            in679 : in std_logic;
            in680 : in std_logic;
            in681 : in std_logic;
            in682 : in std_logic;
            in683 : in std_logic;
            in684 : in std_logic;
            in685 : in std_logic;
            in686 : in std_logic;
            in687 : in std_logic;
            in688 : in std_logic;
            in689 : in std_logic;
            in690 : in std_logic;
            in691 : in std_logic;
            in692 : in std_logic;
            in693 : in std_logic;
            in694 : in std_logic;
            in695 : in std_logic;
            in696 : in std_logic;
            in697 : in std_logic;
            in698 : in std_logic;
            in699 : in std_logic;
            in700 : in std_logic;
            in701 : in std_logic;
            in702 : in std_logic;
            in703 : in std_logic;
            in704 : in std_logic;
            in705 : in std_logic;
            in706 : in std_logic;
            in707 : in std_logic;
            in708 : in std_logic;
            in709 : in std_logic;
            in710 : in std_logic;
            in711 : in std_logic;
            in712 : in std_logic;
            in713 : in std_logic;
            in714 : in std_logic;
            in715 : in std_logic;
            in716 : in std_logic;
            in717 : in std_logic;
            in718 : in std_logic;
            in719 : in std_logic;
            in720 : in std_logic;
            in721 : in std_logic;
            in722 : in std_logic;
            in723 : in std_logic;
            in724 : in std_logic;
            in725 : in std_logic;
            in726 : in std_logic;
            in727 : in std_logic;
            in728 : in std_logic;
            in729 : in std_logic;
            in730 : in std_logic;
            in731 : in std_logic;
            in732 : in std_logic;
            in733 : in std_logic;
            in734 : in std_logic;
            in735 : in std_logic;
            in736 : in std_logic;
            in737 : in std_logic;
            in738 : in std_logic;
            in739 : in std_logic;
            in740 : in std_logic;
            in741 : in std_logic;
            in742 : in std_logic;
            in743 : in std_logic;
            in744 : in std_logic;
            in745 : in std_logic;
            in746 : in std_logic;
            in747 : in std_logic;
            in748 : in std_logic;
            in749 : in std_logic;
            in750 : in std_logic;
            in751 : in std_logic;
            in752 : in std_logic;
            in753 : in std_logic;
            in754 : in std_logic;
            in755 : in std_logic;
            in756 : in std_logic;
            in757 : in std_logic;
            in758 : in std_logic;
            in759 : in std_logic;
            in760 : in std_logic;
            in761 : in std_logic;
            in762 : in std_logic;
            in763 : in std_logic;
            in764 : in std_logic;
            in765 : in std_logic;
            in766 : in std_logic;
            in767 : in std_logic;
            in768 : in std_logic;
            in769 : in std_logic;
            in770 : in std_logic;
            in771 : in std_logic;
            in772 : in std_logic;
            in773 : in std_logic;
            in774 : in std_logic;
            in775 : in std_logic;
            in776 : in std_logic;
            in777 : in std_logic;
            in778 : in std_logic;
            in779 : in std_logic;
            in780 : in std_logic;
            in781 : in std_logic;
            in782 : in std_logic;
            in783 : in std_logic;
            in784 : in std_logic;
            in785 : in std_logic;
            in786 : in std_logic;
            in787 : in std_logic;
            in788 : in std_logic;
            in789 : in std_logic;
            in790 : in std_logic;
            in791 : in std_logic;
            in792 : in std_logic;
            in793 : in std_logic;
            in794 : in std_logic;
            in795 : in std_logic;
            in796 : in std_logic;
            in797 : in std_logic;
            in798 : in std_logic;
            in799 : in std_logic;
            in800 : in std_logic;
            in801 : in std_logic;
            in802 : in std_logic;
            in803 : in std_logic;
            in804 : in std_logic;
            in805 : in std_logic;
            in806 : in std_logic;
            in807 : in std_logic;
            in808 : in std_logic;
            in809 : in std_logic;
            in810 : in std_logic;
            in811 : in std_logic;
            in812 : in std_logic;
            in813 : in std_logic;
            in814 : in std_logic;
            in815 : in std_logic;
            in816 : in std_logic;
            in817 : in std_logic;
            in818 : in std_logic;
            in819 : in std_logic;
            in820 : in std_logic;
            in821 : in std_logic;
            in822 : in std_logic;
            in823 : in std_logic;
            in824 : in std_logic;
            in825 : in std_logic;
            in826 : in std_logic;
            in827 : in std_logic;
            in828 : in std_logic;
            in829 : in std_logic;
            in830 : in std_logic;
            in831 : in std_logic;
            in832 : in std_logic;
            in833 : in std_logic;
            in834 : in std_logic;
            in835 : in std_logic;
            in836 : in std_logic;
            in837 : in std_logic;
            in838 : in std_logic;
            in839 : in std_logic;
            in840 : in std_logic;
            in841 : in std_logic;
            in842 : in std_logic;
            in843 : in std_logic;
            in844 : in std_logic;
            in845 : in std_logic;
            in846 : in std_logic;
            in847 : in std_logic;
            in848 : in std_logic;
            in849 : in std_logic;
            in850 : in std_logic;
            in851 : in std_logic;
            in852 : in std_logic;
            in853 : in std_logic;
            in854 : in std_logic;
            in855 : in std_logic;
            in856 : in std_logic;
            in857 : in std_logic;
            in858 : in std_logic;
            in859 : in std_logic;
            in860 : in std_logic;
            in861 : in std_logic;
            in862 : in std_logic;
            in863 : in std_logic;
            in864 : in std_logic;
            in865 : in std_logic;
            in866 : in std_logic;
            in867 : in std_logic;
            in868 : in std_logic;
            in869 : in std_logic;
            in870 : in std_logic;
            in871 : in std_logic;
            in872 : in std_logic;
            in873 : in std_logic;
            in874 : in std_logic;
            in875 : in std_logic;
            in876 : in std_logic;
            in877 : in std_logic;
            in878 : in std_logic;
            in879 : in std_logic;
            in880 : in std_logic;
            in881 : in std_logic;
            in882 : in std_logic;
            in883 : in std_logic;
            in884 : in std_logic;
            in885 : in std_logic;
            in886 : in std_logic;
            in887 : in std_logic;
            in888 : in std_logic;
            in889 : in std_logic;
            in890 : in std_logic;
            in891 : in std_logic;
            in892 : in std_logic;
            in893 : in std_logic;
            in894 : in std_logic;
            in895 : in std_logic;
            in896 : in std_logic;
            in897 : in std_logic;
            in898 : in std_logic;
            in899 : in std_logic;
            in900 : in std_logic;
            in901 : in std_logic;
            in902 : in std_logic;
            in903 : in std_logic;
            in904 : in std_logic;
            in905 : in std_logic;
            in906 : in std_logic;
            in907 : in std_logic;
            in908 : in std_logic;
            in909 : in std_logic;
            in910 : in std_logic;
            in911 : in std_logic;
            in912 : in std_logic;
            in913 : in std_logic;
            in914 : in std_logic;
            in915 : in std_logic;
            in916 : in std_logic;
            in917 : in std_logic;
            in918 : in std_logic;
            in919 : in std_logic;
            in920 : in std_logic;
            in921 : in std_logic;
            in922 : in std_logic;
            in923 : in std_logic;
            in924 : in std_logic;
            in925 : in std_logic;
            in926 : in std_logic;
            in927 : in std_logic;
            in928 : in std_logic;
            in929 : in std_logic;
            in930 : in std_logic;
            in931 : in std_logic;
            in932 : in std_logic;
            in933 : in std_logic;
            in934 : in std_logic;
            in935 : in std_logic;
            in936 : in std_logic;
            in937 : in std_logic;
            in938 : in std_logic;
            in939 : in std_logic;
            in940 : in std_logic;
            in941 : in std_logic;
            in942 : in std_logic;
            in943 : in std_logic;
            in944 : in std_logic;
            in945 : in std_logic;
            in946 : in std_logic;
            in947 : in std_logic;
            in948 : in std_logic;
            in949 : in std_logic;
            in950 : in std_logic;
            in951 : in std_logic;
            in952 : in std_logic;
            in953 : in std_logic;
            in954 : in std_logic;
            in955 : in std_logic;
            in956 : in std_logic;
            in957 : in std_logic;
            in958 : in std_logic;
            in959 : in std_logic;
            in960 : in std_logic;
            in961 : in std_logic;
            in962 : in std_logic;
            in963 : in std_logic;
            in964 : in std_logic;
            in965 : in std_logic;
            in966 : in std_logic;
            in967 : in std_logic;
            in968 : in std_logic;
            in969 : in std_logic;
            in970 : in std_logic;
            in971 : in std_logic;
            in972 : in std_logic;
            in973 : in std_logic;
            in974 : in std_logic;
            in975 : in std_logic;
            in976 : in std_logic;
            in977 : in std_logic;
            in978 : in std_logic;
            in979 : in std_logic;
            in980 : in std_logic;
            in981 : in std_logic;
            in982 : in std_logic;
            in983 : in std_logic;
            in984 : in std_logic;
            in985 : in std_logic;
            in986 : in std_logic;
            in987 : in std_logic;
            in988 : in std_logic;
            in989 : in std_logic;
            in990 : in std_logic;
            in991 : in std_logic;
            in992 : in std_logic;
            in993 : in std_logic;
            in994 : in std_logic;
            in995 : in std_logic;
            in996 : in std_logic;
            in997 : in std_logic;
            in998 : in std_logic;
            in999 : in std_logic;
            in1000 : in std_logic;
            in1001 : in std_logic;
            in1002 : in std_logic;
            in1003 : in std_logic;
            in1004 : in std_logic;
            in1005 : in std_logic;
            in1006 : in std_logic;
            in1007 : in std_logic;
            in1008 : in std_logic;
            in1009 : in std_logic;
            in1010 : in std_logic;
            in1011 : in std_logic;
            in1012 : in std_logic;
            in1013 : in std_logic;
            in1014 : in std_logic;
            in1015 : in std_logic;
            in1016 : in std_logic;
            in1017 : in std_logic;
            in1018 : in std_logic;
            in1019 : in std_logic;
            in1020 : in std_logic;
            in1021 : in std_logic;
            in1022 : in std_logic;
            in1023 : in std_logic;
            mux_out : out std_logic
        );
    end component;

    component mux_512to1 is
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

    exc_mux : mux_1024to1
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
            in128 => exc_spikes_sampled(128),
            in129 => exc_spikes_sampled(129),
            in130 => exc_spikes_sampled(130),
            in131 => exc_spikes_sampled(131),
            in132 => exc_spikes_sampled(132),
            in133 => exc_spikes_sampled(133),
            in134 => exc_spikes_sampled(134),
            in135 => exc_spikes_sampled(135),
            in136 => exc_spikes_sampled(136),
            in137 => exc_spikes_sampled(137),
            in138 => exc_spikes_sampled(138),
            in139 => exc_spikes_sampled(139),
            in140 => exc_spikes_sampled(140),
            in141 => exc_spikes_sampled(141),
            in142 => exc_spikes_sampled(142),
            in143 => exc_spikes_sampled(143),
            in144 => exc_spikes_sampled(144),
            in145 => exc_spikes_sampled(145),
            in146 => exc_spikes_sampled(146),
            in147 => exc_spikes_sampled(147),
            in148 => exc_spikes_sampled(148),
            in149 => exc_spikes_sampled(149),
            in150 => exc_spikes_sampled(150),
            in151 => exc_spikes_sampled(151),
            in152 => exc_spikes_sampled(152),
            in153 => exc_spikes_sampled(153),
            in154 => exc_spikes_sampled(154),
            in155 => exc_spikes_sampled(155),
            in156 => exc_spikes_sampled(156),
            in157 => exc_spikes_sampled(157),
            in158 => exc_spikes_sampled(158),
            in159 => exc_spikes_sampled(159),
            in160 => exc_spikes_sampled(160),
            in161 => exc_spikes_sampled(161),
            in162 => exc_spikes_sampled(162),
            in163 => exc_spikes_sampled(163),
            in164 => exc_spikes_sampled(164),
            in165 => exc_spikes_sampled(165),
            in166 => exc_spikes_sampled(166),
            in167 => exc_spikes_sampled(167),
            in168 => exc_spikes_sampled(168),
            in169 => exc_spikes_sampled(169),
            in170 => exc_spikes_sampled(170),
            in171 => exc_spikes_sampled(171),
            in172 => exc_spikes_sampled(172),
            in173 => exc_spikes_sampled(173),
            in174 => exc_spikes_sampled(174),
            in175 => exc_spikes_sampled(175),
            in176 => exc_spikes_sampled(176),
            in177 => exc_spikes_sampled(177),
            in178 => exc_spikes_sampled(178),
            in179 => exc_spikes_sampled(179),
            in180 => exc_spikes_sampled(180),
            in181 => exc_spikes_sampled(181),
            in182 => exc_spikes_sampled(182),
            in183 => exc_spikes_sampled(183),
            in184 => exc_spikes_sampled(184),
            in185 => exc_spikes_sampled(185),
            in186 => exc_spikes_sampled(186),
            in187 => exc_spikes_sampled(187),
            in188 => exc_spikes_sampled(188),
            in189 => exc_spikes_sampled(189),
            in190 => exc_spikes_sampled(190),
            in191 => exc_spikes_sampled(191),
            in192 => exc_spikes_sampled(192),
            in193 => exc_spikes_sampled(193),
            in194 => exc_spikes_sampled(194),
            in195 => exc_spikes_sampled(195),
            in196 => exc_spikes_sampled(196),
            in197 => exc_spikes_sampled(197),
            in198 => exc_spikes_sampled(198),
            in199 => exc_spikes_sampled(199),
            in200 => exc_spikes_sampled(200),
            in201 => exc_spikes_sampled(201),
            in202 => exc_spikes_sampled(202),
            in203 => exc_spikes_sampled(203),
            in204 => exc_spikes_sampled(204),
            in205 => exc_spikes_sampled(205),
            in206 => exc_spikes_sampled(206),
            in207 => exc_spikes_sampled(207),
            in208 => exc_spikes_sampled(208),
            in209 => exc_spikes_sampled(209),
            in210 => exc_spikes_sampled(210),
            in211 => exc_spikes_sampled(211),
            in212 => exc_spikes_sampled(212),
            in213 => exc_spikes_sampled(213),
            in214 => exc_spikes_sampled(214),
            in215 => exc_spikes_sampled(215),
            in216 => exc_spikes_sampled(216),
            in217 => exc_spikes_sampled(217),
            in218 => exc_spikes_sampled(218),
            in219 => exc_spikes_sampled(219),
            in220 => exc_spikes_sampled(220),
            in221 => exc_spikes_sampled(221),
            in222 => exc_spikes_sampled(222),
            in223 => exc_spikes_sampled(223),
            in224 => exc_spikes_sampled(224),
            in225 => exc_spikes_sampled(225),
            in226 => exc_spikes_sampled(226),
            in227 => exc_spikes_sampled(227),
            in228 => exc_spikes_sampled(228),
            in229 => exc_spikes_sampled(229),
            in230 => exc_spikes_sampled(230),
            in231 => exc_spikes_sampled(231),
            in232 => exc_spikes_sampled(232),
            in233 => exc_spikes_sampled(233),
            in234 => exc_spikes_sampled(234),
            in235 => exc_spikes_sampled(235),
            in236 => exc_spikes_sampled(236),
            in237 => exc_spikes_sampled(237),
            in238 => exc_spikes_sampled(238),
            in239 => exc_spikes_sampled(239),
            in240 => exc_spikes_sampled(240),
            in241 => exc_spikes_sampled(241),
            in242 => exc_spikes_sampled(242),
            in243 => exc_spikes_sampled(243),
            in244 => exc_spikes_sampled(244),
            in245 => exc_spikes_sampled(245),
            in246 => exc_spikes_sampled(246),
            in247 => exc_spikes_sampled(247),
            in248 => exc_spikes_sampled(248),
            in249 => exc_spikes_sampled(249),
            in250 => exc_spikes_sampled(250),
            in251 => exc_spikes_sampled(251),
            in252 => exc_spikes_sampled(252),
            in253 => exc_spikes_sampled(253),
            in254 => exc_spikes_sampled(254),
            in255 => exc_spikes_sampled(255),
            in256 => exc_spikes_sampled(256),
            in257 => exc_spikes_sampled(257),
            in258 => exc_spikes_sampled(258),
            in259 => exc_spikes_sampled(259),
            in260 => exc_spikes_sampled(260),
            in261 => exc_spikes_sampled(261),
            in262 => exc_spikes_sampled(262),
            in263 => exc_spikes_sampled(263),
            in264 => exc_spikes_sampled(264),
            in265 => exc_spikes_sampled(265),
            in266 => exc_spikes_sampled(266),
            in267 => exc_spikes_sampled(267),
            in268 => exc_spikes_sampled(268),
            in269 => exc_spikes_sampled(269),
            in270 => exc_spikes_sampled(270),
            in271 => exc_spikes_sampled(271),
            in272 => exc_spikes_sampled(272),
            in273 => exc_spikes_sampled(273),
            in274 => exc_spikes_sampled(274),
            in275 => exc_spikes_sampled(275),
            in276 => exc_spikes_sampled(276),
            in277 => exc_spikes_sampled(277),
            in278 => exc_spikes_sampled(278),
            in279 => exc_spikes_sampled(279),
            in280 => exc_spikes_sampled(280),
            in281 => exc_spikes_sampled(281),
            in282 => exc_spikes_sampled(282),
            in283 => exc_spikes_sampled(283),
            in284 => exc_spikes_sampled(284),
            in285 => exc_spikes_sampled(285),
            in286 => exc_spikes_sampled(286),
            in287 => exc_spikes_sampled(287),
            in288 => exc_spikes_sampled(288),
            in289 => exc_spikes_sampled(289),
            in290 => exc_spikes_sampled(290),
            in291 => exc_spikes_sampled(291),
            in292 => exc_spikes_sampled(292),
            in293 => exc_spikes_sampled(293),
            in294 => exc_spikes_sampled(294),
            in295 => exc_spikes_sampled(295),
            in296 => exc_spikes_sampled(296),
            in297 => exc_spikes_sampled(297),
            in298 => exc_spikes_sampled(298),
            in299 => exc_spikes_sampled(299),
            in300 => exc_spikes_sampled(300),
            in301 => exc_spikes_sampled(301),
            in302 => exc_spikes_sampled(302),
            in303 => exc_spikes_sampled(303),
            in304 => exc_spikes_sampled(304),
            in305 => exc_spikes_sampled(305),
            in306 => exc_spikes_sampled(306),
            in307 => exc_spikes_sampled(307),
            in308 => exc_spikes_sampled(308),
            in309 => exc_spikes_sampled(309),
            in310 => exc_spikes_sampled(310),
            in311 => exc_spikes_sampled(311),
            in312 => exc_spikes_sampled(312),
            in313 => exc_spikes_sampled(313),
            in314 => exc_spikes_sampled(314),
            in315 => exc_spikes_sampled(315),
            in316 => exc_spikes_sampled(316),
            in317 => exc_spikes_sampled(317),
            in318 => exc_spikes_sampled(318),
            in319 => exc_spikes_sampled(319),
            in320 => exc_spikes_sampled(320),
            in321 => exc_spikes_sampled(321),
            in322 => exc_spikes_sampled(322),
            in323 => exc_spikes_sampled(323),
            in324 => exc_spikes_sampled(324),
            in325 => exc_spikes_sampled(325),
            in326 => exc_spikes_sampled(326),
            in327 => exc_spikes_sampled(327),
            in328 => exc_spikes_sampled(328),
            in329 => exc_spikes_sampled(329),
            in330 => exc_spikes_sampled(330),
            in331 => exc_spikes_sampled(331),
            in332 => exc_spikes_sampled(332),
            in333 => exc_spikes_sampled(333),
            in334 => exc_spikes_sampled(334),
            in335 => exc_spikes_sampled(335),
            in336 => exc_spikes_sampled(336),
            in337 => exc_spikes_sampled(337),
            in338 => exc_spikes_sampled(338),
            in339 => exc_spikes_sampled(339),
            in340 => exc_spikes_sampled(340),
            in341 => exc_spikes_sampled(341),
            in342 => exc_spikes_sampled(342),
            in343 => exc_spikes_sampled(343),
            in344 => exc_spikes_sampled(344),
            in345 => exc_spikes_sampled(345),
            in346 => exc_spikes_sampled(346),
            in347 => exc_spikes_sampled(347),
            in348 => exc_spikes_sampled(348),
            in349 => exc_spikes_sampled(349),
            in350 => exc_spikes_sampled(350),
            in351 => exc_spikes_sampled(351),
            in352 => exc_spikes_sampled(352),
            in353 => exc_spikes_sampled(353),
            in354 => exc_spikes_sampled(354),
            in355 => exc_spikes_sampled(355),
            in356 => exc_spikes_sampled(356),
            in357 => exc_spikes_sampled(357),
            in358 => exc_spikes_sampled(358),
            in359 => exc_spikes_sampled(359),
            in360 => exc_spikes_sampled(360),
            in361 => exc_spikes_sampled(361),
            in362 => exc_spikes_sampled(362),
            in363 => exc_spikes_sampled(363),
            in364 => exc_spikes_sampled(364),
            in365 => exc_spikes_sampled(365),
            in366 => exc_spikes_sampled(366),
            in367 => exc_spikes_sampled(367),
            in368 => exc_spikes_sampled(368),
            in369 => exc_spikes_sampled(369),
            in370 => exc_spikes_sampled(370),
            in371 => exc_spikes_sampled(371),
            in372 => exc_spikes_sampled(372),
            in373 => exc_spikes_sampled(373),
            in374 => exc_spikes_sampled(374),
            in375 => exc_spikes_sampled(375),
            in376 => exc_spikes_sampled(376),
            in377 => exc_spikes_sampled(377),
            in378 => exc_spikes_sampled(378),
            in379 => exc_spikes_sampled(379),
            in380 => exc_spikes_sampled(380),
            in381 => exc_spikes_sampled(381),
            in382 => exc_spikes_sampled(382),
            in383 => exc_spikes_sampled(383),
            in384 => exc_spikes_sampled(384),
            in385 => exc_spikes_sampled(385),
            in386 => exc_spikes_sampled(386),
            in387 => exc_spikes_sampled(387),
            in388 => exc_spikes_sampled(388),
            in389 => exc_spikes_sampled(389),
            in390 => exc_spikes_sampled(390),
            in391 => exc_spikes_sampled(391),
            in392 => exc_spikes_sampled(392),
            in393 => exc_spikes_sampled(393),
            in394 => exc_spikes_sampled(394),
            in395 => exc_spikes_sampled(395),
            in396 => exc_spikes_sampled(396),
            in397 => exc_spikes_sampled(397),
            in398 => exc_spikes_sampled(398),
            in399 => exc_spikes_sampled(399),
            in400 => exc_spikes_sampled(400),
            in401 => exc_spikes_sampled(401),
            in402 => exc_spikes_sampled(402),
            in403 => exc_spikes_sampled(403),
            in404 => exc_spikes_sampled(404),
            in405 => exc_spikes_sampled(405),
            in406 => exc_spikes_sampled(406),
            in407 => exc_spikes_sampled(407),
            in408 => exc_spikes_sampled(408),
            in409 => exc_spikes_sampled(409),
            in410 => exc_spikes_sampled(410),
            in411 => exc_spikes_sampled(411),
            in412 => exc_spikes_sampled(412),
            in413 => exc_spikes_sampled(413),
            in414 => exc_spikes_sampled(414),
            in415 => exc_spikes_sampled(415),
            in416 => exc_spikes_sampled(416),
            in417 => exc_spikes_sampled(417),
            in418 => exc_spikes_sampled(418),
            in419 => exc_spikes_sampled(419),
            in420 => exc_spikes_sampled(420),
            in421 => exc_spikes_sampled(421),
            in422 => exc_spikes_sampled(422),
            in423 => exc_spikes_sampled(423),
            in424 => exc_spikes_sampled(424),
            in425 => exc_spikes_sampled(425),
            in426 => exc_spikes_sampled(426),
            in427 => exc_spikes_sampled(427),
            in428 => exc_spikes_sampled(428),
            in429 => exc_spikes_sampled(429),
            in430 => exc_spikes_sampled(430),
            in431 => exc_spikes_sampled(431),
            in432 => exc_spikes_sampled(432),
            in433 => exc_spikes_sampled(433),
            in434 => exc_spikes_sampled(434),
            in435 => exc_spikes_sampled(435),
            in436 => exc_spikes_sampled(436),
            in437 => exc_spikes_sampled(437),
            in438 => exc_spikes_sampled(438),
            in439 => exc_spikes_sampled(439),
            in440 => exc_spikes_sampled(440),
            in441 => exc_spikes_sampled(441),
            in442 => exc_spikes_sampled(442),
            in443 => exc_spikes_sampled(443),
            in444 => exc_spikes_sampled(444),
            in445 => exc_spikes_sampled(445),
            in446 => exc_spikes_sampled(446),
            in447 => exc_spikes_sampled(447),
            in448 => exc_spikes_sampled(448),
            in449 => exc_spikes_sampled(449),
            in450 => exc_spikes_sampled(450),
            in451 => exc_spikes_sampled(451),
            in452 => exc_spikes_sampled(452),
            in453 => exc_spikes_sampled(453),
            in454 => exc_spikes_sampled(454),
            in455 => exc_spikes_sampled(455),
            in456 => exc_spikes_sampled(456),
            in457 => exc_spikes_sampled(457),
            in458 => exc_spikes_sampled(458),
            in459 => exc_spikes_sampled(459),
            in460 => exc_spikes_sampled(460),
            in461 => exc_spikes_sampled(461),
            in462 => exc_spikes_sampled(462),
            in463 => exc_spikes_sampled(463),
            in464 => exc_spikes_sampled(464),
            in465 => exc_spikes_sampled(465),
            in466 => exc_spikes_sampled(466),
            in467 => exc_spikes_sampled(467),
            in468 => exc_spikes_sampled(468),
            in469 => exc_spikes_sampled(469),
            in470 => exc_spikes_sampled(470),
            in471 => exc_spikes_sampled(471),
            in472 => exc_spikes_sampled(472),
            in473 => exc_spikes_sampled(473),
            in474 => exc_spikes_sampled(474),
            in475 => exc_spikes_sampled(475),
            in476 => exc_spikes_sampled(476),
            in477 => exc_spikes_sampled(477),
            in478 => exc_spikes_sampled(478),
            in479 => exc_spikes_sampled(479),
            in480 => exc_spikes_sampled(480),
            in481 => exc_spikes_sampled(481),
            in482 => exc_spikes_sampled(482),
            in483 => exc_spikes_sampled(483),
            in484 => exc_spikes_sampled(484),
            in485 => exc_spikes_sampled(485),
            in486 => exc_spikes_sampled(486),
            in487 => exc_spikes_sampled(487),
            in488 => exc_spikes_sampled(488),
            in489 => exc_spikes_sampled(489),
            in490 => exc_spikes_sampled(490),
            in491 => exc_spikes_sampled(491),
            in492 => exc_spikes_sampled(492),
            in493 => exc_spikes_sampled(493),
            in494 => exc_spikes_sampled(494),
            in495 => exc_spikes_sampled(495),
            in496 => exc_spikes_sampled(496),
            in497 => exc_spikes_sampled(497),
            in498 => exc_spikes_sampled(498),
            in499 => exc_spikes_sampled(499),
            in500 => exc_spikes_sampled(500),
            in501 => exc_spikes_sampled(501),
            in502 => exc_spikes_sampled(502),
            in503 => exc_spikes_sampled(503),
            in504 => exc_spikes_sampled(504),
            in505 => exc_spikes_sampled(505),
            in506 => exc_spikes_sampled(506),
            in507 => exc_spikes_sampled(507),
            in508 => exc_spikes_sampled(508),
            in509 => exc_spikes_sampled(509),
            in510 => exc_spikes_sampled(510),
            in511 => exc_spikes_sampled(511),
            in512 => exc_spikes_sampled(512),
            in513 => exc_spikes_sampled(513),
            in514 => exc_spikes_sampled(514),
            in515 => exc_spikes_sampled(515),
            in516 => exc_spikes_sampled(516),
            in517 => exc_spikes_sampled(517),
            in518 => exc_spikes_sampled(518),
            in519 => exc_spikes_sampled(519),
            in520 => exc_spikes_sampled(520),
            in521 => exc_spikes_sampled(521),
            in522 => exc_spikes_sampled(522),
            in523 => exc_spikes_sampled(523),
            in524 => exc_spikes_sampled(524),
            in525 => exc_spikes_sampled(525),
            in526 => exc_spikes_sampled(526),
            in527 => exc_spikes_sampled(527),
            in528 => exc_spikes_sampled(528),
            in529 => exc_spikes_sampled(529),
            in530 => exc_spikes_sampled(530),
            in531 => exc_spikes_sampled(531),
            in532 => exc_spikes_sampled(532),
            in533 => exc_spikes_sampled(533),
            in534 => exc_spikes_sampled(534),
            in535 => exc_spikes_sampled(535),
            in536 => exc_spikes_sampled(536),
            in537 => exc_spikes_sampled(537),
            in538 => exc_spikes_sampled(538),
            in539 => exc_spikes_sampled(539),
            in540 => exc_spikes_sampled(540),
            in541 => exc_spikes_sampled(541),
            in542 => exc_spikes_sampled(542),
            in543 => exc_spikes_sampled(543),
            in544 => exc_spikes_sampled(544),
            in545 => exc_spikes_sampled(545),
            in546 => exc_spikes_sampled(546),
            in547 => exc_spikes_sampled(547),
            in548 => exc_spikes_sampled(548),
            in549 => exc_spikes_sampled(549),
            in550 => exc_spikes_sampled(550),
            in551 => exc_spikes_sampled(551),
            in552 => exc_spikes_sampled(552),
            in553 => exc_spikes_sampled(553),
            in554 => exc_spikes_sampled(554),
            in555 => exc_spikes_sampled(555),
            in556 => exc_spikes_sampled(556),
            in557 => exc_spikes_sampled(557),
            in558 => exc_spikes_sampled(558),
            in559 => exc_spikes_sampled(559),
            in560 => exc_spikes_sampled(560),
            in561 => exc_spikes_sampled(561),
            in562 => exc_spikes_sampled(562),
            in563 => exc_spikes_sampled(563),
            in564 => exc_spikes_sampled(564),
            in565 => exc_spikes_sampled(565),
            in566 => exc_spikes_sampled(566),
            in567 => exc_spikes_sampled(567),
            in568 => exc_spikes_sampled(568),
            in569 => exc_spikes_sampled(569),
            in570 => exc_spikes_sampled(570),
            in571 => exc_spikes_sampled(571),
            in572 => exc_spikes_sampled(572),
            in573 => exc_spikes_sampled(573),
            in574 => exc_spikes_sampled(574),
            in575 => exc_spikes_sampled(575),
            in576 => exc_spikes_sampled(576),
            in577 => exc_spikes_sampled(577),
            in578 => exc_spikes_sampled(578),
            in579 => exc_spikes_sampled(579),
            in580 => exc_spikes_sampled(580),
            in581 => exc_spikes_sampled(581),
            in582 => exc_spikes_sampled(582),
            in583 => exc_spikes_sampled(583),
            in584 => exc_spikes_sampled(584),
            in585 => exc_spikes_sampled(585),
            in586 => exc_spikes_sampled(586),
            in587 => exc_spikes_sampled(587),
            in588 => exc_spikes_sampled(588),
            in589 => exc_spikes_sampled(589),
            in590 => exc_spikes_sampled(590),
            in591 => exc_spikes_sampled(591),
            in592 => exc_spikes_sampled(592),
            in593 => exc_spikes_sampled(593),
            in594 => exc_spikes_sampled(594),
            in595 => exc_spikes_sampled(595),
            in596 => exc_spikes_sampled(596),
            in597 => exc_spikes_sampled(597),
            in598 => exc_spikes_sampled(598),
            in599 => exc_spikes_sampled(599),
            in600 => exc_spikes_sampled(600),
            in601 => exc_spikes_sampled(601),
            in602 => exc_spikes_sampled(602),
            in603 => exc_spikes_sampled(603),
            in604 => exc_spikes_sampled(604),
            in605 => exc_spikes_sampled(605),
            in606 => exc_spikes_sampled(606),
            in607 => exc_spikes_sampled(607),
            in608 => exc_spikes_sampled(608),
            in609 => exc_spikes_sampled(609),
            in610 => exc_spikes_sampled(610),
            in611 => exc_spikes_sampled(611),
            in612 => exc_spikes_sampled(612),
            in613 => exc_spikes_sampled(613),
            in614 => exc_spikes_sampled(614),
            in615 => exc_spikes_sampled(615),
            in616 => exc_spikes_sampled(616),
            in617 => exc_spikes_sampled(617),
            in618 => exc_spikes_sampled(618),
            in619 => exc_spikes_sampled(619),
            in620 => exc_spikes_sampled(620),
            in621 => exc_spikes_sampled(621),
            in622 => exc_spikes_sampled(622),
            in623 => exc_spikes_sampled(623),
            in624 => exc_spikes_sampled(624),
            in625 => exc_spikes_sampled(625),
            in626 => exc_spikes_sampled(626),
            in627 => exc_spikes_sampled(627),
            in628 => exc_spikes_sampled(628),
            in629 => exc_spikes_sampled(629),
            in630 => exc_spikes_sampled(630),
            in631 => exc_spikes_sampled(631),
            in632 => exc_spikes_sampled(632),
            in633 => exc_spikes_sampled(633),
            in634 => exc_spikes_sampled(634),
            in635 => exc_spikes_sampled(635),
            in636 => exc_spikes_sampled(636),
            in637 => exc_spikes_sampled(637),
            in638 => exc_spikes_sampled(638),
            in639 => exc_spikes_sampled(639),
            in640 => exc_spikes_sampled(640),
            in641 => exc_spikes_sampled(641),
            in642 => exc_spikes_sampled(642),
            in643 => exc_spikes_sampled(643),
            in644 => exc_spikes_sampled(644),
            in645 => exc_spikes_sampled(645),
            in646 => exc_spikes_sampled(646),
            in647 => exc_spikes_sampled(647),
            in648 => exc_spikes_sampled(648),
            in649 => exc_spikes_sampled(649),
            in650 => exc_spikes_sampled(650),
            in651 => exc_spikes_sampled(651),
            in652 => exc_spikes_sampled(652),
            in653 => exc_spikes_sampled(653),
            in654 => exc_spikes_sampled(654),
            in655 => exc_spikes_sampled(655),
            in656 => exc_spikes_sampled(656),
            in657 => exc_spikes_sampled(657),
            in658 => exc_spikes_sampled(658),
            in659 => exc_spikes_sampled(659),
            in660 => exc_spikes_sampled(660),
            in661 => exc_spikes_sampled(661),
            in662 => exc_spikes_sampled(662),
            in663 => exc_spikes_sampled(663),
            in664 => exc_spikes_sampled(664),
            in665 => exc_spikes_sampled(665),
            in666 => exc_spikes_sampled(666),
            in667 => exc_spikes_sampled(667),
            in668 => exc_spikes_sampled(668),
            in669 => exc_spikes_sampled(669),
            in670 => exc_spikes_sampled(670),
            in671 => exc_spikes_sampled(671),
            in672 => exc_spikes_sampled(672),
            in673 => exc_spikes_sampled(673),
            in674 => exc_spikes_sampled(674),
            in675 => exc_spikes_sampled(675),
            in676 => exc_spikes_sampled(676),
            in677 => exc_spikes_sampled(677),
            in678 => exc_spikes_sampled(678),
            in679 => exc_spikes_sampled(679),
            in680 => exc_spikes_sampled(680),
            in681 => exc_spikes_sampled(681),
            in682 => exc_spikes_sampled(682),
            in683 => exc_spikes_sampled(683),
            in684 => exc_spikes_sampled(684),
            in685 => exc_spikes_sampled(685),
            in686 => exc_spikes_sampled(686),
            in687 => exc_spikes_sampled(687),
            in688 => exc_spikes_sampled(688),
            in689 => exc_spikes_sampled(689),
            in690 => exc_spikes_sampled(690),
            in691 => exc_spikes_sampled(691),
            in692 => exc_spikes_sampled(692),
            in693 => exc_spikes_sampled(693),
            in694 => exc_spikes_sampled(694),
            in695 => exc_spikes_sampled(695),
            in696 => exc_spikes_sampled(696),
            in697 => exc_spikes_sampled(697),
            in698 => exc_spikes_sampled(698),
            in699 => exc_spikes_sampled(699),
            in700 => exc_spikes_sampled(700),
            in701 => exc_spikes_sampled(701),
            in702 => exc_spikes_sampled(702),
            in703 => exc_spikes_sampled(703),
            in704 => exc_spikes_sampled(704),
            in705 => exc_spikes_sampled(705),
            in706 => exc_spikes_sampled(706),
            in707 => exc_spikes_sampled(707),
            in708 => exc_spikes_sampled(708),
            in709 => exc_spikes_sampled(709),
            in710 => exc_spikes_sampled(710),
            in711 => exc_spikes_sampled(711),
            in712 => exc_spikes_sampled(712),
            in713 => exc_spikes_sampled(713),
            in714 => exc_spikes_sampled(714),
            in715 => exc_spikes_sampled(715),
            in716 => exc_spikes_sampled(716),
            in717 => exc_spikes_sampled(717),
            in718 => exc_spikes_sampled(718),
            in719 => exc_spikes_sampled(719),
            in720 => exc_spikes_sampled(720),
            in721 => exc_spikes_sampled(721),
            in722 => exc_spikes_sampled(722),
            in723 => exc_spikes_sampled(723),
            in724 => exc_spikes_sampled(724),
            in725 => exc_spikes_sampled(725),
            in726 => exc_spikes_sampled(726),
            in727 => exc_spikes_sampled(727),
            in728 => exc_spikes_sampled(728),
            in729 => exc_spikes_sampled(729),
            in730 => exc_spikes_sampled(730),
            in731 => exc_spikes_sampled(731),
            in732 => exc_spikes_sampled(732),
            in733 => exc_spikes_sampled(733),
            in734 => exc_spikes_sampled(734),
            in735 => exc_spikes_sampled(735),
            in736 => exc_spikes_sampled(736),
            in737 => exc_spikes_sampled(737),
            in738 => exc_spikes_sampled(738),
            in739 => exc_spikes_sampled(739),
            in740 => exc_spikes_sampled(740),
            in741 => exc_spikes_sampled(741),
            in742 => exc_spikes_sampled(742),
            in743 => exc_spikes_sampled(743),
            in744 => exc_spikes_sampled(744),
            in745 => exc_spikes_sampled(745),
            in746 => exc_spikes_sampled(746),
            in747 => exc_spikes_sampled(747),
            in748 => exc_spikes_sampled(748),
            in749 => exc_spikes_sampled(749),
            in750 => exc_spikes_sampled(750),
            in751 => exc_spikes_sampled(751),
            in752 => exc_spikes_sampled(752),
            in753 => exc_spikes_sampled(753),
            in754 => exc_spikes_sampled(754),
            in755 => exc_spikes_sampled(755),
            in756 => exc_spikes_sampled(756),
            in757 => exc_spikes_sampled(757),
            in758 => exc_spikes_sampled(758),
            in759 => exc_spikes_sampled(759),
            in760 => exc_spikes_sampled(760),
            in761 => exc_spikes_sampled(761),
            in762 => exc_spikes_sampled(762),
            in763 => exc_spikes_sampled(763),
            in764 => exc_spikes_sampled(764),
            in765 => exc_spikes_sampled(765),
            in766 => exc_spikes_sampled(766),
            in767 => exc_spikes_sampled(767),
            in768 => exc_spikes_sampled(768),
            in769 => exc_spikes_sampled(769),
            in770 => exc_spikes_sampled(770),
            in771 => exc_spikes_sampled(771),
            in772 => exc_spikes_sampled(772),
            in773 => exc_spikes_sampled(773),
            in774 => exc_spikes_sampled(774),
            in775 => exc_spikes_sampled(775),
            in776 => exc_spikes_sampled(776),
            in777 => exc_spikes_sampled(777),
            in778 => exc_spikes_sampled(778),
            in779 => exc_spikes_sampled(779),
            in780 => exc_spikes_sampled(780),
            in781 => exc_spikes_sampled(781),
            in782 => exc_spikes_sampled(782),
            in783 => exc_spikes_sampled(783),
            in784 => exc_spikes_sampled(784),
            in785 => exc_spikes_sampled(785),
            in786 => exc_spikes_sampled(786),
            in787 => exc_spikes_sampled(787),
            in788 => exc_spikes_sampled(788),
            in789 => exc_spikes_sampled(789),
            in790 => exc_spikes_sampled(790),
            in791 => exc_spikes_sampled(791),
            in792 => exc_spikes_sampled(792),
            in793 => exc_spikes_sampled(793),
            in794 => exc_spikes_sampled(794),
            in795 => exc_spikes_sampled(795),
            in796 => exc_spikes_sampled(796),
            in797 => exc_spikes_sampled(797),
            in798 => exc_spikes_sampled(798),
            in799 => exc_spikes_sampled(799),
            in800 => exc_spikes_sampled(800),
            in801 => exc_spikes_sampled(801),
            in802 => exc_spikes_sampled(802),
            in803 => exc_spikes_sampled(803),
            in804 => exc_spikes_sampled(804),
            in805 => exc_spikes_sampled(805),
            in806 => exc_spikes_sampled(806),
            in807 => exc_spikes_sampled(807),
            in808 => exc_spikes_sampled(808),
            in809 => exc_spikes_sampled(809),
            in810 => exc_spikes_sampled(810),
            in811 => exc_spikes_sampled(811),
            in812 => exc_spikes_sampled(812),
            in813 => exc_spikes_sampled(813),
            in814 => exc_spikes_sampled(814),
            in815 => exc_spikes_sampled(815),
            in816 => exc_spikes_sampled(816),
            in817 => exc_spikes_sampled(817),
            in818 => exc_spikes_sampled(818),
            in819 => exc_spikes_sampled(819),
            in820 => exc_spikes_sampled(820),
            in821 => exc_spikes_sampled(821),
            in822 => exc_spikes_sampled(822),
            in823 => exc_spikes_sampled(823),
            in824 => exc_spikes_sampled(824),
            in825 => exc_spikes_sampled(825),
            in826 => exc_spikes_sampled(826),
            in827 => exc_spikes_sampled(827),
            in828 => exc_spikes_sampled(828),
            in829 => exc_spikes_sampled(829),
            in830 => exc_spikes_sampled(830),
            in831 => exc_spikes_sampled(831),
            in832 => exc_spikes_sampled(832),
            in833 => exc_spikes_sampled(833),
            in834 => exc_spikes_sampled(834),
            in835 => exc_spikes_sampled(835),
            in836 => exc_spikes_sampled(836),
            in837 => exc_spikes_sampled(837),
            in838 => exc_spikes_sampled(838),
            in839 => exc_spikes_sampled(839),
            in840 => exc_spikes_sampled(840),
            in841 => exc_spikes_sampled(841),
            in842 => exc_spikes_sampled(842),
            in843 => exc_spikes_sampled(843),
            in844 => exc_spikes_sampled(844),
            in845 => exc_spikes_sampled(845),
            in846 => exc_spikes_sampled(846),
            in847 => exc_spikes_sampled(847),
            in848 => exc_spikes_sampled(848),
            in849 => exc_spikes_sampled(849),
            in850 => exc_spikes_sampled(850),
            in851 => exc_spikes_sampled(851),
            in852 => exc_spikes_sampled(852),
            in853 => exc_spikes_sampled(853),
            in854 => exc_spikes_sampled(854),
            in855 => exc_spikes_sampled(855),
            in856 => exc_spikes_sampled(856),
            in857 => exc_spikes_sampled(857),
            in858 => exc_spikes_sampled(858),
            in859 => exc_spikes_sampled(859),
            in860 => exc_spikes_sampled(860),
            in861 => exc_spikes_sampled(861),
            in862 => exc_spikes_sampled(862),
            in863 => exc_spikes_sampled(863),
            in864 => exc_spikes_sampled(864),
            in865 => exc_spikes_sampled(865),
            in866 => exc_spikes_sampled(866),
            in867 => exc_spikes_sampled(867),
            in868 => exc_spikes_sampled(868),
            in869 => exc_spikes_sampled(869),
            in870 => exc_spikes_sampled(870),
            in871 => exc_spikes_sampled(871),
            in872 => exc_spikes_sampled(872),
            in873 => exc_spikes_sampled(873),
            in874 => exc_spikes_sampled(874),
            in875 => exc_spikes_sampled(875),
            in876 => exc_spikes_sampled(876),
            in877 => exc_spikes_sampled(877),
            in878 => exc_spikes_sampled(878),
            in879 => exc_spikes_sampled(879),
            in880 => exc_spikes_sampled(880),
            in881 => exc_spikes_sampled(881),
            in882 => exc_spikes_sampled(882),
            in883 => exc_spikes_sampled(883),
            in884 => exc_spikes_sampled(884),
            in885 => exc_spikes_sampled(885),
            in886 => exc_spikes_sampled(886),
            in887 => exc_spikes_sampled(887),
            in888 => exc_spikes_sampled(888),
            in889 => exc_spikes_sampled(889),
            in890 => exc_spikes_sampled(890),
            in891 => exc_spikes_sampled(891),
            in892 => exc_spikes_sampled(892),
            in893 => exc_spikes_sampled(893),
            in894 => exc_spikes_sampled(894),
            in895 => exc_spikes_sampled(895),
            in896 => exc_spikes_sampled(896),
            in897 => exc_spikes_sampled(897),
            in898 => exc_spikes_sampled(898),
            in899 => exc_spikes_sampled(899),
            in900 => exc_spikes_sampled(900),
            in901 => exc_spikes_sampled(901),
            in902 => exc_spikes_sampled(902),
            in903 => exc_spikes_sampled(903),
            in904 => exc_spikes_sampled(904),
            in905 => exc_spikes_sampled(905),
            in906 => exc_spikes_sampled(906),
            in907 => exc_spikes_sampled(907),
            in908 => exc_spikes_sampled(908),
            in909 => exc_spikes_sampled(909),
            in910 => exc_spikes_sampled(910),
            in911 => exc_spikes_sampled(911),
            in912 => exc_spikes_sampled(912),
            in913 => exc_spikes_sampled(913),
            in914 => exc_spikes_sampled(914),
            in915 => exc_spikes_sampled(915),
            in916 => exc_spikes_sampled(916),
            in917 => exc_spikes_sampled(917),
            in918 => exc_spikes_sampled(918),
            in919 => exc_spikes_sampled(919),
            in920 => exc_spikes_sampled(920),
            in921 => exc_spikes_sampled(921),
            in922 => exc_spikes_sampled(922),
            in923 => exc_spikes_sampled(923),
            in924 => exc_spikes_sampled(924),
            in925 => exc_spikes_sampled(925),
            in926 => exc_spikes_sampled(926),
            in927 => exc_spikes_sampled(927),
            in928 => exc_spikes_sampled(928),
            in929 => exc_spikes_sampled(929),
            in930 => exc_spikes_sampled(930),
            in931 => exc_spikes_sampled(931),
            in932 => exc_spikes_sampled(932),
            in933 => exc_spikes_sampled(933),
            in934 => exc_spikes_sampled(934),
            in935 => exc_spikes_sampled(935),
            in936 => exc_spikes_sampled(936),
            in937 => exc_spikes_sampled(937),
            in938 => exc_spikes_sampled(938),
            in939 => exc_spikes_sampled(939),
            in940 => exc_spikes_sampled(940),
            in941 => exc_spikes_sampled(941),
            in942 => exc_spikes_sampled(942),
            in943 => exc_spikes_sampled(943),
            in944 => exc_spikes_sampled(944),
            in945 => exc_spikes_sampled(945),
            in946 => exc_spikes_sampled(946),
            in947 => exc_spikes_sampled(947),
            in948 => exc_spikes_sampled(948),
            in949 => exc_spikes_sampled(949),
            in950 => exc_spikes_sampled(950),
            in951 => exc_spikes_sampled(951),
            in952 => exc_spikes_sampled(952),
            in953 => exc_spikes_sampled(953),
            in954 => exc_spikes_sampled(954),
            in955 => exc_spikes_sampled(955),
            in956 => exc_spikes_sampled(956),
            in957 => exc_spikes_sampled(957),
            in958 => exc_spikes_sampled(958),
            in959 => exc_spikes_sampled(959),
            in960 => exc_spikes_sampled(960),
            in961 => exc_spikes_sampled(961),
            in962 => exc_spikes_sampled(962),
            in963 => exc_spikes_sampled(963),
            in964 => exc_spikes_sampled(964),
            in965 => exc_spikes_sampled(965),
            in966 => exc_spikes_sampled(966),
            in967 => exc_spikes_sampled(967),
            in968 => exc_spikes_sampled(968),
            in969 => exc_spikes_sampled(969),
            in970 => exc_spikes_sampled(970),
            in971 => exc_spikes_sampled(971),
            in972 => exc_spikes_sampled(972),
            in973 => exc_spikes_sampled(973),
            in974 => exc_spikes_sampled(974),
            in975 => exc_spikes_sampled(975),
            in976 => exc_spikes_sampled(976),
            in977 => exc_spikes_sampled(977),
            in978 => exc_spikes_sampled(978),
            in979 => exc_spikes_sampled(979),
            in980 => exc_spikes_sampled(980),
            in981 => exc_spikes_sampled(981),
            in982 => exc_spikes_sampled(982),
            in983 => exc_spikes_sampled(983),
            in984 => exc_spikes_sampled(984),
            in985 => exc_spikes_sampled(985),
            in986 => exc_spikes_sampled(986),
            in987 => exc_spikes_sampled(987),
            in988 => exc_spikes_sampled(988),
            in989 => exc_spikes_sampled(989),
            in990 => exc_spikes_sampled(990),
            in991 => exc_spikes_sampled(991),
            in992 => exc_spikes_sampled(992),
            in993 => exc_spikes_sampled(993),
            in994 => exc_spikes_sampled(994),
            in995 => exc_spikes_sampled(995),
            in996 => exc_spikes_sampled(996),
            in997 => exc_spikes_sampled(997),
            in998 => exc_spikes_sampled(998),
            in999 => exc_spikes_sampled(999),
            in1000 => exc_spikes_sampled(1000),
            in1001 => exc_spikes_sampled(1001),
            in1002 => exc_spikes_sampled(1002),
            in1003 => exc_spikes_sampled(1003),
            in1004 => exc_spikes_sampled(1004),
            in1005 => exc_spikes_sampled(1005),
            in1006 => exc_spikes_sampled(1006),
            in1007 => exc_spikes_sampled(1007),
            in1008 => exc_spikes_sampled(1008),
            in1009 => exc_spikes_sampled(1009),
            in1010 => exc_spikes_sampled(1010),
            in1011 => exc_spikes_sampled(1011),
            in1012 => exc_spikes_sampled(1012),
            in1013 => exc_spikes_sampled(1013),
            in1014 => exc_spikes_sampled(1014),
            in1015 => exc_spikes_sampled(1015),
            in1016 => exc_spikes_sampled(1016),
            in1017 => exc_spikes_sampled(1017),
            in1018 => exc_spikes_sampled(1018),
            in1019 => exc_spikes_sampled(1019),
            in1020 => exc_spikes_sampled(1020),
            in1021 => exc_spikes_sampled(1021),
            in1022 => exc_spikes_sampled(1022),
            in1023 => exc_spikes_sampled(1023),
            mux_out => exc_spike
        );

    inh_mux : mux_512to1
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
            in10 => inh_spikes_sampled(10),
            in11 => inh_spikes_sampled(11),
            in12 => inh_spikes_sampled(12),
            in13 => inh_spikes_sampled(13),
            in14 => inh_spikes_sampled(14),
            in15 => inh_spikes_sampled(15),
            in16 => inh_spikes_sampled(16),
            in17 => inh_spikes_sampled(17),
            in18 => inh_spikes_sampled(18),
            in19 => inh_spikes_sampled(19),
            in20 => inh_spikes_sampled(20),
            in21 => inh_spikes_sampled(21),
            in22 => inh_spikes_sampled(22),
            in23 => inh_spikes_sampled(23),
            in24 => inh_spikes_sampled(24),
            in25 => inh_spikes_sampled(25),
            in26 => inh_spikes_sampled(26),
            in27 => inh_spikes_sampled(27),
            in28 => inh_spikes_sampled(28),
            in29 => inh_spikes_sampled(29),
            in30 => inh_spikes_sampled(30),
            in31 => inh_spikes_sampled(31),
            in32 => inh_spikes_sampled(32),
            in33 => inh_spikes_sampled(33),
            in34 => inh_spikes_sampled(34),
            in35 => inh_spikes_sampled(35),
            in36 => inh_spikes_sampled(36),
            in37 => inh_spikes_sampled(37),
            in38 => inh_spikes_sampled(38),
            in39 => inh_spikes_sampled(39),
            in40 => inh_spikes_sampled(40),
            in41 => inh_spikes_sampled(41),
            in42 => inh_spikes_sampled(42),
            in43 => inh_spikes_sampled(43),
            in44 => inh_spikes_sampled(44),
            in45 => inh_spikes_sampled(45),
            in46 => inh_spikes_sampled(46),
            in47 => inh_spikes_sampled(47),
            in48 => inh_spikes_sampled(48),
            in49 => inh_spikes_sampled(49),
            in50 => inh_spikes_sampled(50),
            in51 => inh_spikes_sampled(51),
            in52 => inh_spikes_sampled(52),
            in53 => inh_spikes_sampled(53),
            in54 => inh_spikes_sampled(54),
            in55 => inh_spikes_sampled(55),
            in56 => inh_spikes_sampled(56),
            in57 => inh_spikes_sampled(57),
            in58 => inh_spikes_sampled(58),
            in59 => inh_spikes_sampled(59),
            in60 => inh_spikes_sampled(60),
            in61 => inh_spikes_sampled(61),
            in62 => inh_spikes_sampled(62),
            in63 => inh_spikes_sampled(63),
            in64 => inh_spikes_sampled(64),
            in65 => inh_spikes_sampled(65),
            in66 => inh_spikes_sampled(66),
            in67 => inh_spikes_sampled(67),
            in68 => inh_spikes_sampled(68),
            in69 => inh_spikes_sampled(69),
            in70 => inh_spikes_sampled(70),
            in71 => inh_spikes_sampled(71),
            in72 => inh_spikes_sampled(72),
            in73 => inh_spikes_sampled(73),
            in74 => inh_spikes_sampled(74),
            in75 => inh_spikes_sampled(75),
            in76 => inh_spikes_sampled(76),
            in77 => inh_spikes_sampled(77),
            in78 => inh_spikes_sampled(78),
            in79 => inh_spikes_sampled(79),
            in80 => inh_spikes_sampled(80),
            in81 => inh_spikes_sampled(81),
            in82 => inh_spikes_sampled(82),
            in83 => inh_spikes_sampled(83),
            in84 => inh_spikes_sampled(84),
            in85 => inh_spikes_sampled(85),
            in86 => inh_spikes_sampled(86),
            in87 => inh_spikes_sampled(87),
            in88 => inh_spikes_sampled(88),
            in89 => inh_spikes_sampled(89),
            in90 => inh_spikes_sampled(90),
            in91 => inh_spikes_sampled(91),
            in92 => inh_spikes_sampled(92),
            in93 => inh_spikes_sampled(93),
            in94 => inh_spikes_sampled(94),
            in95 => inh_spikes_sampled(95),
            in96 => inh_spikes_sampled(96),
            in97 => inh_spikes_sampled(97),
            in98 => inh_spikes_sampled(98),
            in99 => inh_spikes_sampled(99),
            in100 => inh_spikes_sampled(100),
            in101 => inh_spikes_sampled(101),
            in102 => inh_spikes_sampled(102),
            in103 => inh_spikes_sampled(103),
            in104 => inh_spikes_sampled(104),
            in105 => inh_spikes_sampled(105),
            in106 => inh_spikes_sampled(106),
            in107 => inh_spikes_sampled(107),
            in108 => inh_spikes_sampled(108),
            in109 => inh_spikes_sampled(109),
            in110 => inh_spikes_sampled(110),
            in111 => inh_spikes_sampled(111),
            in112 => inh_spikes_sampled(112),
            in113 => inh_spikes_sampled(113),
            in114 => inh_spikes_sampled(114),
            in115 => inh_spikes_sampled(115),
            in116 => inh_spikes_sampled(116),
            in117 => inh_spikes_sampled(117),
            in118 => inh_spikes_sampled(118),
            in119 => inh_spikes_sampled(119),
            in120 => inh_spikes_sampled(120),
            in121 => inh_spikes_sampled(121),
            in122 => inh_spikes_sampled(122),
            in123 => inh_spikes_sampled(123),
            in124 => inh_spikes_sampled(124),
            in125 => inh_spikes_sampled(125),
            in126 => inh_spikes_sampled(126),
            in127 => inh_spikes_sampled(127),
            in128 => inh_spikes_sampled(128),
            in129 => inh_spikes_sampled(129),
            in130 => inh_spikes_sampled(130),
            in131 => inh_spikes_sampled(131),
            in132 => inh_spikes_sampled(132),
            in133 => inh_spikes_sampled(133),
            in134 => inh_spikes_sampled(134),
            in135 => inh_spikes_sampled(135),
            in136 => inh_spikes_sampled(136),
            in137 => inh_spikes_sampled(137),
            in138 => inh_spikes_sampled(138),
            in139 => inh_spikes_sampled(139),
            in140 => inh_spikes_sampled(140),
            in141 => inh_spikes_sampled(141),
            in142 => inh_spikes_sampled(142),
            in143 => inh_spikes_sampled(143),
            in144 => inh_spikes_sampled(144),
            in145 => inh_spikes_sampled(145),
            in146 => inh_spikes_sampled(146),
            in147 => inh_spikes_sampled(147),
            in148 => inh_spikes_sampled(148),
            in149 => inh_spikes_sampled(149),
            in150 => inh_spikes_sampled(150),
            in151 => inh_spikes_sampled(151),
            in152 => inh_spikes_sampled(152),
            in153 => inh_spikes_sampled(153),
            in154 => inh_spikes_sampled(154),
            in155 => inh_spikes_sampled(155),
            in156 => inh_spikes_sampled(156),
            in157 => inh_spikes_sampled(157),
            in158 => inh_spikes_sampled(158),
            in159 => inh_spikes_sampled(159),
            in160 => inh_spikes_sampled(160),
            in161 => inh_spikes_sampled(161),
            in162 => inh_spikes_sampled(162),
            in163 => inh_spikes_sampled(163),
            in164 => inh_spikes_sampled(164),
            in165 => inh_spikes_sampled(165),
            in166 => inh_spikes_sampled(166),
            in167 => inh_spikes_sampled(167),
            in168 => inh_spikes_sampled(168),
            in169 => inh_spikes_sampled(169),
            in170 => inh_spikes_sampled(170),
            in171 => inh_spikes_sampled(171),
            in172 => inh_spikes_sampled(172),
            in173 => inh_spikes_sampled(173),
            in174 => inh_spikes_sampled(174),
            in175 => inh_spikes_sampled(175),
            in176 => inh_spikes_sampled(176),
            in177 => inh_spikes_sampled(177),
            in178 => inh_spikes_sampled(178),
            in179 => inh_spikes_sampled(179),
            in180 => inh_spikes_sampled(180),
            in181 => inh_spikes_sampled(181),
            in182 => inh_spikes_sampled(182),
            in183 => inh_spikes_sampled(183),
            in184 => inh_spikes_sampled(184),
            in185 => inh_spikes_sampled(185),
            in186 => inh_spikes_sampled(186),
            in187 => inh_spikes_sampled(187),
            in188 => inh_spikes_sampled(188),
            in189 => inh_spikes_sampled(189),
            in190 => inh_spikes_sampled(190),
            in191 => inh_spikes_sampled(191),
            in192 => inh_spikes_sampled(192),
            in193 => inh_spikes_sampled(193),
            in194 => inh_spikes_sampled(194),
            in195 => inh_spikes_sampled(195),
            in196 => inh_spikes_sampled(196),
            in197 => inh_spikes_sampled(197),
            in198 => inh_spikes_sampled(198),
            in199 => inh_spikes_sampled(199),
            in200 => inh_spikes_sampled(200),
            in201 => inh_spikes_sampled(201),
            in202 => inh_spikes_sampled(202),
            in203 => inh_spikes_sampled(203),
            in204 => inh_spikes_sampled(204),
            in205 => inh_spikes_sampled(205),
            in206 => inh_spikes_sampled(206),
            in207 => inh_spikes_sampled(207),
            in208 => inh_spikes_sampled(208),
            in209 => inh_spikes_sampled(209),
            in210 => inh_spikes_sampled(210),
            in211 => inh_spikes_sampled(211),
            in212 => inh_spikes_sampled(212),
            in213 => inh_spikes_sampled(213),
            in214 => inh_spikes_sampled(214),
            in215 => inh_spikes_sampled(215),
            in216 => inh_spikes_sampled(216),
            in217 => inh_spikes_sampled(217),
            in218 => inh_spikes_sampled(218),
            in219 => inh_spikes_sampled(219),
            in220 => inh_spikes_sampled(220),
            in221 => inh_spikes_sampled(221),
            in222 => inh_spikes_sampled(222),
            in223 => inh_spikes_sampled(223),
            in224 => inh_spikes_sampled(224),
            in225 => inh_spikes_sampled(225),
            in226 => inh_spikes_sampled(226),
            in227 => inh_spikes_sampled(227),
            in228 => inh_spikes_sampled(228),
            in229 => inh_spikes_sampled(229),
            in230 => inh_spikes_sampled(230),
            in231 => inh_spikes_sampled(231),
            in232 => inh_spikes_sampled(232),
            in233 => inh_spikes_sampled(233),
            in234 => inh_spikes_sampled(234),
            in235 => inh_spikes_sampled(235),
            in236 => inh_spikes_sampled(236),
            in237 => inh_spikes_sampled(237),
            in238 => inh_spikes_sampled(238),
            in239 => inh_spikes_sampled(239),
            in240 => inh_spikes_sampled(240),
            in241 => inh_spikes_sampled(241),
            in242 => inh_spikes_sampled(242),
            in243 => inh_spikes_sampled(243),
            in244 => inh_spikes_sampled(244),
            in245 => inh_spikes_sampled(245),
            in246 => inh_spikes_sampled(246),
            in247 => inh_spikes_sampled(247),
            in248 => inh_spikes_sampled(248),
            in249 => inh_spikes_sampled(249),
            in250 => inh_spikes_sampled(250),
            in251 => inh_spikes_sampled(251),
            in252 => inh_spikes_sampled(252),
            in253 => inh_spikes_sampled(253),
            in254 => inh_spikes_sampled(254),
            in255 => inh_spikes_sampled(255),
            in256 => inh_spikes_sampled(256),
            in257 => inh_spikes_sampled(257),
            in258 => inh_spikes_sampled(258),
            in259 => inh_spikes_sampled(259),
            in260 => inh_spikes_sampled(260),
            in261 => inh_spikes_sampled(261),
            in262 => inh_spikes_sampled(262),
            in263 => inh_spikes_sampled(263),
            in264 => inh_spikes_sampled(264),
            in265 => inh_spikes_sampled(265),
            in266 => inh_spikes_sampled(266),
            in267 => inh_spikes_sampled(267),
            in268 => inh_spikes_sampled(268),
            in269 => inh_spikes_sampled(269),
            in270 => inh_spikes_sampled(270),
            in271 => inh_spikes_sampled(271),
            in272 => inh_spikes_sampled(272),
            in273 => inh_spikes_sampled(273),
            in274 => inh_spikes_sampled(274),
            in275 => inh_spikes_sampled(275),
            in276 => inh_spikes_sampled(276),
            in277 => inh_spikes_sampled(277),
            in278 => inh_spikes_sampled(278),
            in279 => inh_spikes_sampled(279),
            in280 => inh_spikes_sampled(280),
            in281 => inh_spikes_sampled(281),
            in282 => inh_spikes_sampled(282),
            in283 => inh_spikes_sampled(283),
            in284 => inh_spikes_sampled(284),
            in285 => inh_spikes_sampled(285),
            in286 => inh_spikes_sampled(286),
            in287 => inh_spikes_sampled(287),
            in288 => inh_spikes_sampled(288),
            in289 => inh_spikes_sampled(289),
            in290 => inh_spikes_sampled(290),
            in291 => inh_spikes_sampled(291),
            in292 => inh_spikes_sampled(292),
            in293 => inh_spikes_sampled(293),
            in294 => inh_spikes_sampled(294),
            in295 => inh_spikes_sampled(295),
            in296 => inh_spikes_sampled(296),
            in297 => inh_spikes_sampled(297),
            in298 => inh_spikes_sampled(298),
            in299 => inh_spikes_sampled(299),
            in300 => inh_spikes_sampled(300),
            in301 => inh_spikes_sampled(301),
            in302 => inh_spikes_sampled(302),
            in303 => inh_spikes_sampled(303),
            in304 => inh_spikes_sampled(304),
            in305 => inh_spikes_sampled(305),
            in306 => inh_spikes_sampled(306),
            in307 => inh_spikes_sampled(307),
            in308 => inh_spikes_sampled(308),
            in309 => inh_spikes_sampled(309),
            in310 => inh_spikes_sampled(310),
            in311 => inh_spikes_sampled(311),
            in312 => inh_spikes_sampled(312),
            in313 => inh_spikes_sampled(313),
            in314 => inh_spikes_sampled(314),
            in315 => inh_spikes_sampled(315),
            in316 => inh_spikes_sampled(316),
            in317 => inh_spikes_sampled(317),
            in318 => inh_spikes_sampled(318),
            in319 => inh_spikes_sampled(319),
            in320 => inh_spikes_sampled(320),
            in321 => inh_spikes_sampled(321),
            in322 => inh_spikes_sampled(322),
            in323 => inh_spikes_sampled(323),
            in324 => inh_spikes_sampled(324),
            in325 => inh_spikes_sampled(325),
            in326 => inh_spikes_sampled(326),
            in327 => inh_spikes_sampled(327),
            in328 => inh_spikes_sampled(328),
            in329 => inh_spikes_sampled(329),
            in330 => inh_spikes_sampled(330),
            in331 => inh_spikes_sampled(331),
            in332 => inh_spikes_sampled(332),
            in333 => inh_spikes_sampled(333),
            in334 => inh_spikes_sampled(334),
            in335 => inh_spikes_sampled(335),
            in336 => inh_spikes_sampled(336),
            in337 => inh_spikes_sampled(337),
            in338 => inh_spikes_sampled(338),
            in339 => inh_spikes_sampled(339),
            in340 => inh_spikes_sampled(340),
            in341 => inh_spikes_sampled(341),
            in342 => inh_spikes_sampled(342),
            in343 => inh_spikes_sampled(343),
            in344 => inh_spikes_sampled(344),
            in345 => inh_spikes_sampled(345),
            in346 => inh_spikes_sampled(346),
            in347 => inh_spikes_sampled(347),
            in348 => inh_spikes_sampled(348),
            in349 => inh_spikes_sampled(349),
            in350 => inh_spikes_sampled(350),
            in351 => inh_spikes_sampled(351),
            in352 => inh_spikes_sampled(352),
            in353 => inh_spikes_sampled(353),
            in354 => inh_spikes_sampled(354),
            in355 => inh_spikes_sampled(355),
            in356 => inh_spikes_sampled(356),
            in357 => inh_spikes_sampled(357),
            in358 => inh_spikes_sampled(358),
            in359 => inh_spikes_sampled(359),
            in360 => inh_spikes_sampled(360),
            in361 => inh_spikes_sampled(361),
            in362 => inh_spikes_sampled(362),
            in363 => inh_spikes_sampled(363),
            in364 => inh_spikes_sampled(364),
            in365 => inh_spikes_sampled(365),
            in366 => inh_spikes_sampled(366),
            in367 => inh_spikes_sampled(367),
            in368 => inh_spikes_sampled(368),
            in369 => inh_spikes_sampled(369),
            in370 => inh_spikes_sampled(370),
            in371 => inh_spikes_sampled(371),
            in372 => inh_spikes_sampled(372),
            in373 => inh_spikes_sampled(373),
            in374 => inh_spikes_sampled(374),
            in375 => inh_spikes_sampled(375),
            in376 => inh_spikes_sampled(376),
            in377 => inh_spikes_sampled(377),
            in378 => inh_spikes_sampled(378),
            in379 => inh_spikes_sampled(379),
            in380 => inh_spikes_sampled(380),
            in381 => inh_spikes_sampled(381),
            in382 => inh_spikes_sampled(382),
            in383 => inh_spikes_sampled(383),
            in384 => inh_spikes_sampled(384),
            in385 => inh_spikes_sampled(385),
            in386 => inh_spikes_sampled(386),
            in387 => inh_spikes_sampled(387),
            in388 => inh_spikes_sampled(388),
            in389 => inh_spikes_sampled(389),
            in390 => inh_spikes_sampled(390),
            in391 => inh_spikes_sampled(391),
            in392 => inh_spikes_sampled(392),
            in393 => inh_spikes_sampled(393),
            in394 => inh_spikes_sampled(394),
            in395 => inh_spikes_sampled(395),
            in396 => inh_spikes_sampled(396),
            in397 => inh_spikes_sampled(397),
            in398 => inh_spikes_sampled(398),
            in399 => inh_spikes_sampled(399),
            in400 => inh_spikes_sampled(400),
            in401 => inh_spikes_sampled(401),
            in402 => inh_spikes_sampled(402),
            in403 => inh_spikes_sampled(403),
            in404 => inh_spikes_sampled(404),
            in405 => inh_spikes_sampled(405),
            in406 => inh_spikes_sampled(406),
            in407 => inh_spikes_sampled(407),
            in408 => inh_spikes_sampled(408),
            in409 => inh_spikes_sampled(409),
            in410 => inh_spikes_sampled(410),
            in411 => inh_spikes_sampled(411),
            in412 => inh_spikes_sampled(412),
            in413 => inh_spikes_sampled(413),
            in414 => inh_spikes_sampled(414),
            in415 => inh_spikes_sampled(415),
            in416 => inh_spikes_sampled(416),
            in417 => inh_spikes_sampled(417),
            in418 => inh_spikes_sampled(418),
            in419 => inh_spikes_sampled(419),
            in420 => inh_spikes_sampled(420),
            in421 => inh_spikes_sampled(421),
            in422 => inh_spikes_sampled(422),
            in423 => inh_spikes_sampled(423),
            in424 => inh_spikes_sampled(424),
            in425 => inh_spikes_sampled(425),
            in426 => inh_spikes_sampled(426),
            in427 => inh_spikes_sampled(427),
            in428 => inh_spikes_sampled(428),
            in429 => inh_spikes_sampled(429),
            in430 => inh_spikes_sampled(430),
            in431 => inh_spikes_sampled(431),
            in432 => inh_spikes_sampled(432),
            in433 => inh_spikes_sampled(433),
            in434 => inh_spikes_sampled(434),
            in435 => inh_spikes_sampled(435),
            in436 => inh_spikes_sampled(436),
            in437 => inh_spikes_sampled(437),
            in438 => inh_spikes_sampled(438),
            in439 => inh_spikes_sampled(439),
            in440 => inh_spikes_sampled(440),
            in441 => inh_spikes_sampled(441),
            in442 => inh_spikes_sampled(442),
            in443 => inh_spikes_sampled(443),
            in444 => inh_spikes_sampled(444),
            in445 => inh_spikes_sampled(445),
            in446 => inh_spikes_sampled(446),
            in447 => inh_spikes_sampled(447),
            in448 => inh_spikes_sampled(448),
            in449 => inh_spikes_sampled(449),
            in450 => inh_spikes_sampled(450),
            in451 => inh_spikes_sampled(451),
            in452 => inh_spikes_sampled(452),
            in453 => inh_spikes_sampled(453),
            in454 => inh_spikes_sampled(454),
            in455 => inh_spikes_sampled(455),
            in456 => inh_spikes_sampled(456),
            in457 => inh_spikes_sampled(457),
            in458 => inh_spikes_sampled(458),
            in459 => inh_spikes_sampled(459),
            in460 => inh_spikes_sampled(460),
            in461 => inh_spikes_sampled(461),
            in462 => inh_spikes_sampled(462),
            in463 => inh_spikes_sampled(463),
            in464 => inh_spikes_sampled(464),
            in465 => inh_spikes_sampled(465),
            in466 => inh_spikes_sampled(466),
            in467 => inh_spikes_sampled(467),
            in468 => inh_spikes_sampled(468),
            in469 => inh_spikes_sampled(469),
            in470 => inh_spikes_sampled(470),
            in471 => inh_spikes_sampled(471),
            in472 => inh_spikes_sampled(472),
            in473 => inh_spikes_sampled(473),
            in474 => inh_spikes_sampled(474),
            in475 => inh_spikes_sampled(475),
            in476 => inh_spikes_sampled(476),
            in477 => inh_spikes_sampled(477),
            in478 => inh_spikes_sampled(478),
            in479 => inh_spikes_sampled(479),
            in480 => inh_spikes_sampled(480),
            in481 => inh_spikes_sampled(481),
            in482 => inh_spikes_sampled(482),
            in483 => inh_spikes_sampled(483),
            in484 => inh_spikes_sampled(484),
            in485 => inh_spikes_sampled(485),
            in486 => inh_spikes_sampled(486),
            in487 => inh_spikes_sampled(487),
            in488 => inh_spikes_sampled(488),
            in489 => inh_spikes_sampled(489),
            in490 => inh_spikes_sampled(490),
            in491 => inh_spikes_sampled(491),
            in492 => inh_spikes_sampled(492),
            in493 => inh_spikes_sampled(493),
            in494 => inh_spikes_sampled(494),
            in495 => inh_spikes_sampled(495),
            in496 => inh_spikes_sampled(496),
            in497 => inh_spikes_sampled(497),
            in498 => inh_spikes_sampled(498),
            in499 => inh_spikes_sampled(499),
            in500 => inh_spikes_sampled(500),
            in501 => inh_spikes_sampled(501),
            in502 => inh_spikes_sampled(502),
            in503 => inh_spikes_sampled(503),
            in504 => inh_spikes_sampled(504),
            in505 => inh_spikes_sampled(505),
            in506 => inh_spikes_sampled(506),
            in507 => inh_spikes_sampled(507),
            in508 => inh_spikes_sampled(508),
            in509 => inh_spikes_sampled(509),
            in510 => inh_spikes_sampled(510),
            in511 => inh_spikes_sampled(511),
            mux_out => inh_spike
        );

    exc_counter : cnt
        generic map(
            N => exc_cnt_bitwidth,
            rst_value => 1023
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
            rst_value => 511
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

