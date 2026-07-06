library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package snn_pkg is
  type int_vec_t is array (natural range <>) of integer;
  type int_matrix_t is array (natural range <>, natural range <>) of integer;

  function sat_int(x : integer; lo : integer; hi : integer) return integer;
  function sign_i(x : integer) return integer;
  function argmax(v : int_vec_t) return integer;
  function get_pixel(img : std_logic_vector; idx : natural) return integer;
  function pseudo_weight(row : natural; col : natural; spread : natural) return integer;
  function pseudo_feedback(row : natural; col : natural) return integer;
end package;

package body snn_pkg is
  function sat_int(x : integer; lo : integer; hi : integer) return integer is
  begin
    if x < lo then
      return lo;
    elsif x > hi then
      return hi;
    else
      return x;
    end if;
  end function;

  function sign_i(x : integer) return integer is
  begin
    if x < 0 then
      return -1;
    elsif x > 0 then
      return 1;
    else
      return 0;
    end if;
  end function;

  function argmax(v : int_vec_t) return integer is
    variable best_i : integer := v'low;
    variable best_v : integer := v(v'low);
  begin
    for i in v'low + 1 to v'high loop
      if v(i) > best_v then
        best_v := v(i);
        best_i := i;
      end if;
    end loop;
    return best_i;
  end function;

  function get_pixel(img : std_logic_vector; idx : natural) return integer is
    variable lo  : natural := idx * 8;
    variable hi  : natural := (idx * 8) + 7;
    variable pix : unsigned(7 downto 0);
  begin
    pix := unsigned(img(hi downto lo));
    return to_integer(pix);
  end function;

  function pseudo_weight(row : natural; col : natural; spread : natural) return integer is
    variable span : integer;
    variable raw  : integer;
  begin
    if spread = 0 then
      return 0;
    end if;
    span := integer((2 * spread) + 1);
    raw  := ((integer(row) * 131) + (integer(col) * 17) + 7) mod span;
    return raw - integer(spread);
  end function;

  function pseudo_feedback(row : natural; col : natural) return integer is
  begin
    if (((integer(row) * 31) + (integer(col) * 7) + 3) mod 2) = 0 then
      return 1;
    else
      return -1;
    end if;
  end function;
end package body;
