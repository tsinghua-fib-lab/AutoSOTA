library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package vgg7_pkg is

  -- ==============================================================
  -- CIFAR-10 input geometry
  -- ==============================================================
  constant C_IMG_CH      : natural := 3;
  constant C_IMG_H       : natural := 32;
  constant C_IMG_W       : natural := 32;
  constant C_IN_PIXELS   : natural := C_IMG_CH * C_IMG_H * C_IMG_W;
  constant C_CLASSES     : natural := 10;

  -- ==============================================================
  -- VGG-7 topology
  -- Exact VGG-small-7 widths often used for CIFAR-10:
  --   128, 128, 256, 256, 512, 512, FC(10)
  -- Default below is a simulation-friendly profile for Vivado XSIM.
  -- Change these six constants to the exact widths if needed.
  -- ==============================================================
  constant C_C1          : natural := 8;
  constant C_C2          : natural := 8;
  constant C_C3          : natural := 16;
  constant C_C4          : natural := 16;
  constant C_C5          : natural := 32;
  constant C_C6          : natural := 32;

  constant C_K           : natural := 3;

  constant C_F1_H        : natural := 32;
  constant C_F1_W        : natural := 32;
  constant C_P1_H        : natural := 16;
  constant C_P1_W        : natural := 16;
  constant C_F3_H        : natural := 16;
  constant C_F3_W        : natural := 16;
  constant C_P2_H        : natural := 8;
  constant C_P2_W        : natural := 8;
  constant C_F5_H        : natural := 8;
  constant C_F5_W        : natural := 8;
  constant C_P3_H        : natural := 4;
  constant C_P3_W        : natural := 4;

  constant C_FMAP0_LEN   : natural := C_IN_PIXELS;
  constant C_FMAP1_LEN   : natural := C_C1 * C_F1_H * C_F1_W;
  constant C_FMAP2_LEN   : natural := C_C2 * C_F1_H * C_F1_W;
  constant C_POOL1_LEN   : natural := C_C2 * C_P1_H * C_P1_W;
  constant C_FMAP3_LEN   : natural := C_C3 * C_F3_H * C_F3_W;
  constant C_FMAP4_LEN   : natural := C_C4 * C_F3_H * C_F3_W;
  constant C_POOL2_LEN   : natural := C_C4 * C_P2_H * C_P2_W;
  constant C_FMAP5_LEN   : natural := C_C5 * C_F5_H * C_F5_W;
  constant C_FMAP6_LEN   : natural := C_C6 * C_F5_H * C_F5_W;
  constant C_POOL3_LEN   : natural := C_C6 * C_P3_H * C_P3_W;
  constant C_FC_IN       : natural := C_POOL3_LEN;

  constant C_W1_LEN      : natural := C_C1 * C_IMG_CH * C_K * C_K;
  constant C_W2_LEN      : natural := C_C2 * C_C1     * C_K * C_K;
  constant C_W3_LEN      : natural := C_C3 * C_C2     * C_K * C_K;
  constant C_W4_LEN      : natural := C_C4 * C_C3     * C_K * C_K;
  constant C_W5_LEN      : natural := C_C5 * C_C4     * C_K * C_K;
  constant C_W6_LEN      : natural := C_C6 * C_C5     * C_K * C_K;
  constant C_W7_LEN      : natural := C_CLASSES * C_FC_IN;

  -- ==============================================================
  -- Numeric configuration
  -- ==============================================================
  constant C_PIX_W       : natural := 8;
  constant C_ACT_W       : natural := 12;
  constant C_WEIGHT_W    : natural := 8;
  constant C_SCORE_W     : natural := 20;
  constant C_LABEL_W     : natural := 4;

  constant C_CONV_DIV_PER_IN_CH : natural := 16;
  constant C_TARGET_SCORE       : integer := 256;

  constant C_LR_CONV     : integer := 1;
  constant C_LR_FC       : integer := 1;

  constant C_USE_ALL_LAYER_UPDATES : boolean := true;

  constant C_W1_SEED     : integer := 11;
  constant C_W2_SEED     : integer := 17;
  constant C_W3_SEED     : integer := 23;
  constant C_W4_SEED     : integer := 29;
  constant C_W5_SEED     : integer := 31;
  constant C_W6_SEED     : integer := 37;
  constant C_W7_SEED     : integer := 41;

  constant C_FB1_SEED    : integer := 101;
  constant C_FB2_SEED    : integer := 103;
  constant C_FB3_SEED    : integer := 107;
  constant C_FB4_SEED    : integer := 109;
  constant C_FB5_SEED    : integer := 113;
  constant C_FB6_SEED    : integer := 127;

  subtype pixel_t  is unsigned(C_PIX_W-1 downto 0);
  subtype act_t    is signed(C_ACT_W-1 downto 0);
  subtype weight_t is signed(C_WEIGHT_W-1 downto 0);
  subtype score_t  is signed(C_SCORE_W-1 downto 0);
  subtype label_t  is unsigned(C_LABEL_W-1 downto 0);
  subtype sign_t   is integer range -1 to 1;

  type pixel_vec_t  is array (natural range <>) of pixel_t;
  type act_vec_t    is array (natural range <>) of act_t;
  type weight_vec_t is array (natural range <>) of weight_t;
  type score_vec_t  is array (natural range <>) of score_t;
  type sign_vec_t   is array (natural range <>) of sign_t;
  type sign_mat_t   is array (natural range <>, natural range <>) of sign_t;

  function idx3(
    c : natural;
    y : natural;
    x : natural;
    h : natural;
    w : natural
  ) return natural;

  function widx4(
    oc   : natural;
    ic   : natural;
    ky   : natural;
    kx   : natural;
    n_ic : natural;
    k_h  : natural;
    k_w  : natural
  ) return natural;

  function fcidx(
    o    : natural;
    i    : natural;
    n_in : natural
  ) return natural;

  function sat_act_signed(a : integer) return act_t;
  function relu_act(a : integer) return act_t;
  function sat_score(a : integer) return score_t;
  function sat_weight_add(a : weight_t; delta : integer) return weight_t;

  function sign_i(a : integer) return sign_t;
  function pixel_to_act(p : pixel_t) return act_t;

  function init_weight_vec(
    n_items   : natural;
    seed_base : integer
  ) return weight_vec_t;

  function init_feedback_matrix(
    n_rows    : natural;
    n_cols    : natural;
    seed_base : integer
  ) return sign_mat_t;

  function argmax_scores(x : score_vec_t) return label_t;

  function pack_pixels(x : pixel_vec_t) return std_logic_vector;
  function unpack_pixels(x : std_logic_vector) return pixel_vec_t;

end package;

package body vgg7_pkg is

  function idx3(
    c : natural;
    y : natural;
    x : natural;
    h : natural;
    w : natural
  ) return natural is
  begin
    return c * h * w + y * w + x;
  end function;

  function widx4(
    oc   : natural;
    ic   : natural;
    ky   : natural;
    kx   : natural;
    n_ic : natural;
    k_h  : natural;
    k_w  : natural
  ) return natural is
  begin
    return (((oc * n_ic) + ic) * k_h + ky) * k_w + kx;
  end function;

  function fcidx(
    o    : natural;
    i    : natural;
    n_in : natural
  ) return natural is
  begin
    return o * n_in + i;
  end function;

  function sat_act_signed(a : integer) return act_t is
    variable v      : integer := a;
    constant A_MAX  : integer := 2**(C_ACT_W-1) - 1;
    constant A_MIN  : integer := -(2**(C_ACT_W-1));
  begin
    if v > A_MAX then
      v := A_MAX;
    elsif v < A_MIN then
      v := A_MIN;
    end if;
    return to_signed(v, C_ACT_W);
  end function;

  function relu_act(a : integer) return act_t is
    variable v      : integer := a;
    constant A_MAX  : integer := 2**(C_ACT_W-1) - 1;
  begin
    if v < 0 then
      v := 0;
    elsif v > A_MAX then
      v := A_MAX;
    end if;
    return to_signed(v, C_ACT_W);
  end function;

  function sat_score(a : integer) return score_t is
    variable v      : integer := a;
    constant S_MAX  : integer := 2**(C_SCORE_W-1) - 1;
    constant S_MIN  : integer := -(2**(C_SCORE_W-1));
  begin
    if v > S_MAX then
      v := S_MAX;
    elsif v < S_MIN then
      v := S_MIN;
    end if;
    return to_signed(v, C_SCORE_W);
  end function;

  function sat_weight_add(a : weight_t; delta : integer) return weight_t is
    variable v      : integer := to_integer(a) + delta;
    constant W_MAX  : integer := 2**(C_WEIGHT_W-1) - 1;
    constant W_MIN  : integer := -(2**(C_WEIGHT_W-1));
  begin
    if v > W_MAX then
      v := W_MAX;
    elsif v < W_MIN then
      v := W_MIN;
    end if;
    return to_signed(v, C_WEIGHT_W);
  end function;

  function sign_i(a : integer) return sign_t is
  begin
    if a > 0 then
      return 1;
    elsif a < 0 then
      return -1;
    else
      return 0;
    end if;
  end function;

  function pixel_to_act(p : pixel_t) return act_t is
  begin
    return sat_act_signed(to_integer(p) - 128);
  end function;

  function init_weight_vec(
    n_items   : natural;
    seed_base : integer
  ) return weight_vec_t is
    variable r : weight_vec_t(0 to n_items-1);
    variable h : integer;
  begin
    for i in 0 to n_items-1 loop
      h := ((i + 1) * 73 + seed_base * 37) mod 5;
      r(i) := to_signed(h - 2, C_WEIGHT_W); -- values in [-2, 2]
    end loop;
    return r;
  end function;

  function init_feedback_matrix(
    n_rows    : natural;
    n_cols    : natural;
    seed_base : integer
  ) return sign_mat_t is
    variable r : sign_mat_t(0 to n_rows-1, 0 to n_cols-1);
    variable h : integer;
  begin
    for row in 0 to n_rows-1 loop
      for col in 0 to n_cols-1 loop
        h := ((row + 3) * 11 + (col + 5) * 17 + seed_base * 19) mod 3;
        case h is
          when 0 =>
            r(row, col) := -1;
          when 1 =>
            r(row, col) := 0;
          when others =>
            r(row, col) := 1;
        end case;
      end loop;
    end loop;
    return r;
  end function;

  function argmax_scores(x : score_vec_t) return label_t is
    variable best_idx : integer := x'low;
    variable best_val : integer := to_integer(x(x'low));
  begin
    for i in x'range loop
      if to_integer(x(i)) > best_val then
        best_val := to_integer(x(i));
        best_idx := i;
      end if;
    end loop;
    return to_unsigned(best_idx, C_LABEL_W);
  end function;

  function pack_pixels(x : pixel_vec_t) return std_logic_vector is
    variable r    : std_logic_vector(x'length * C_PIX_W - 1 downto 0);
    variable base : integer;
  begin
    for i in x'range loop
      base := (i - x'low) * C_PIX_W;
      r(base + C_PIX_W - 1 downto base) := std_logic_vector(x(i));
    end loop;
    return r;
  end function;

  function unpack_pixels(x : std_logic_vector) return pixel_vec_t is
    constant N    : natural := x'length / C_PIX_W;
    variable r    : pixel_vec_t(0 to N-1);
    variable base : integer;
  begin
    for i in 0 to N-1 loop
      base := i * C_PIX_W;
      r(i) := unsigned(x(base + C_PIX_W - 1 downto base));
    end loop;
    return r;
  end function;

end package body;
