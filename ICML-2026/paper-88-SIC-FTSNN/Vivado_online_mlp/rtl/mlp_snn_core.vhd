library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.snn_pkg.all;

entity mlp_snn_core is
  generic(
    G_N_INPUTS             : positive := 784;
    G_N_HIDDEN             : positive := 64;
    G_N_OUTPUTS            : positive := 10;
    G_TIMESTEPS            : positive := 10;
    G_LEAK_SHIFT           : natural  := 4;
    G_THRESHOLD_H          : positive := 16;
    G_THRESHOLD_O          : positive := 8;
    G_W_MIN                : integer  := -7;
    G_W_MAX                : integer  := 7;
    G_INIT_SPREAD          : natural  := 1;
    G_MAX_UPD              : positive := 2;
    G_LR_W1                : positive := 1;
    G_LR_W2                : positive := 1;
    G_HERR_CLIP            : positive := 8;
    G_ENABLE_W1_TRAINING   : boolean  := true;
    G_OUTPUT_RESET_NONE    : boolean  := true
  );
  port(
    clk          : in  std_logic;
    rst          : in  std_logic;
    start_i      : in  std_logic;
    train_i      : in  std_logic;
    image_i      : in  std_logic_vector((G_N_INPUTS * 8) - 1 downto 0);
    label_i      : in  std_logic_vector(3 downto 0);
    ready_o      : out std_logic;
    pred_label_o : out std_logic_vector(3 downto 0);
    correct_o    : out std_logic
  );
end entity;

architecture rtl of mlp_snn_core is
  type state_t is (
    ST_INIT_W1,
    ST_INIT_W2,
    ST_INIT_FB,
    ST_IDLE,
    ST_CLEAR_SAMPLE,
    ST_ENCODE,
    ST_H_ACCUM,
    ST_H_FIRE,
    ST_O_ACCUM,
    ST_O_FIRE,
    ST_PREP_ERR,
    ST_LEARN_W2,
    ST_PREP_HERR,
    ST_LEARN_W1,
    ST_ARGMAX,
    ST_DONE
  );

  signal state      : state_t := ST_INIT_W1;
  signal w1         : int_matrix_t(0 to G_N_HIDDEN - 1, 0 to G_N_INPUTS - 1);
  signal w2         : int_matrix_t(0 to G_N_OUTPUTS - 1, 0 to G_N_HIDDEN - 1);
  signal fb         : int_matrix_t(0 to G_N_HIDDEN - 1, 0 to G_N_OUTPUTS - 1);

  signal mem_h      : int_vec_t(0 to G_N_HIDDEN - 1);
  signal mem_o      : int_vec_t(0 to G_N_OUTPUTS - 1);
  signal cur_h      : int_vec_t(0 to G_N_HIDDEN - 1);
  signal cur_o      : int_vec_t(0 to G_N_OUTPUTS - 1);
  signal cnt_in     : int_vec_t(0 to G_N_INPUTS - 1);
  signal cnt_h      : int_vec_t(0 to G_N_HIDDEN - 1);
  signal cnt_o      : int_vec_t(0 to G_N_OUTPUTS - 1);
  signal out_err    : int_vec_t(0 to G_N_OUTPUTS - 1);
  signal hidden_err : int_vec_t(0 to G_N_HIDDEN - 1);

  signal spike_in   : std_logic_vector(G_N_INPUTS - 1 downto 0) := (others => '0');
  signal spike_h    : std_logic_vector(G_N_HIDDEN - 1 downto 0) := (others => '0');
  signal spike_o    : std_logic_vector(G_N_OUTPUTS - 1 downto 0) := (others => '0');

  signal image_reg  : std_logic_vector((G_N_INPUTS * 8) - 1 downto 0) := (others => '0');
  signal label_reg  : integer range 0 to 15 := 0;
  signal pred_reg   : integer range 0 to 15 := 0;
  signal train_reg  : std_logic := '0';

  signal idx_row    : integer range 0 to 65535 := 0;
  signal idx_col    : integer range 0 to 65535 := 0;
  signal step_idx   : integer range 0 to 65535 := 0;

  signal ready_r    : std_logic := '0';
  signal correct_r  : std_logic := '0';
begin
  ready_o      <= ready_r;
  correct_o    <= correct_r;
  pred_label_o <= std_logic_vector(to_unsigned(pred_reg, pred_label_o'length));

  process(clk)
    variable pix_v      : integer;
    variable thr_v      : integer;
    variable mem_v      : integer;
    variable upd_v      : integer;
    variable pred_v     : integer;
    variable sum_v      : integer;
    variable lab_v      : integer;
    variable leak_div_v : integer;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state     <= ST_INIT_W1;
        idx_row   <= 0;
        idx_col   <= 0;
        step_idx  <= 0;
        ready_r   <= '0';
        correct_r <= '0';
        train_reg <= '0';
        pred_reg  <= 0;
        label_reg <= 0;
        image_reg <= (others => '0');
        spike_in  <= (others => '0');
        spike_h   <= (others => '0');
        spike_o   <= (others => '0');

        for i in 0 to G_N_INPUTS - 1 loop
          cnt_in(i) <= 0;
        end loop;

        for h in 0 to G_N_HIDDEN - 1 loop
          mem_h(h)      <= 0;
          cur_h(h)      <= 0;
          cnt_h(h)      <= 0;
          hidden_err(h) <= 0;
        end loop;

        for o in 0 to G_N_OUTPUTS - 1 loop
          mem_o(o)   <= 0;
          cur_o(o)   <= 0;
          cnt_o(o)   <= 0;
          out_err(o) <= 0;
        end loop;

      else
        ready_r <= '0';

        case state is
          when ST_INIT_W1 =>
            w1(idx_row, idx_col) <= pseudo_weight(idx_row, idx_col, G_INIT_SPREAD);

            if idx_col = G_N_INPUTS - 1 then
              idx_col <= 0;
              if idx_row = G_N_HIDDEN - 1 then
                idx_row <= 0;
                state   <= ST_INIT_W2;
              else
                idx_row <= idx_row + 1;
              end if;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_INIT_W2 =>
            w2(idx_row, idx_col) <= pseudo_weight(idx_row + 13, idx_col + 29, G_INIT_SPREAD);

            if idx_col = G_N_HIDDEN - 1 then
              idx_col <= 0;
              if idx_row = G_N_OUTPUTS - 1 then
                idx_row <= 0;
                state   <= ST_INIT_FB;
              else
                idx_row <= idx_row + 1;
              end if;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_INIT_FB =>
            fb(idx_row, idx_col) <= pseudo_feedback(idx_row, idx_col);

            if idx_col = G_N_OUTPUTS - 1 then
              idx_col <= 0;
              if idx_row = G_N_HIDDEN - 1 then
                idx_row <= 0;
                state   <= ST_IDLE;
              else
                idx_row <= idx_row + 1;
              end if;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_IDLE =>
            ready_r <= '1';

            if start_i = '1' then
              image_reg <= image_i;
              lab_v     := to_integer(unsigned(label_i));
              label_reg <= sat_int(lab_v, 0, G_N_OUTPUTS - 1);
              train_reg <= train_i;
              correct_r <= '0';
              state     <= ST_CLEAR_SAMPLE;
            end if;

          when ST_CLEAR_SAMPLE =>
            step_idx <= 0;
            idx_row  <= 0;
            idx_col  <= 0;

            for i in 0 to G_N_INPUTS - 1 loop
              cnt_in(i)  <= 0;
              spike_in(i) <= '0';
            end loop;

            for h in 0 to G_N_HIDDEN - 1 loop
              mem_h(h)      <= 0;
              cur_h(h)      <= 0;
              cnt_h(h)      <= 0;
              hidden_err(h) <= 0;
              spike_h(h)    <= '0';
            end loop;

            for o in 0 to G_N_OUTPUTS - 1 loop
              mem_o(o)   <= 0;
              cur_o(o)   <= 0;
              cnt_o(o)   <= 0;
              out_err(o) <= 0;
              spike_o(o) <= '0';
            end loop;

            state <= ST_ENCODE;

          when ST_ENCODE =>
            thr_v := (step_idx * 256) / G_TIMESTEPS;

            for i in 0 to G_N_INPUTS - 1 loop
              pix_v := get_pixel(image_reg, i);
              if pix_v > thr_v then
                spike_in(i) <= '1';
                cnt_in(i)   <= cnt_in(i) + 1;
              else
                spike_in(i) <= '0';
              end if;
            end loop;

            for h in 0 to G_N_HIDDEN - 1 loop
              cur_h(h) <= 0;
            end loop;

            idx_col <= 0;
            state   <= ST_H_ACCUM;

          when ST_H_ACCUM =>
            if spike_in(idx_col) = '1' then
              for h in 0 to G_N_HIDDEN - 1 loop
                cur_h(h) <= cur_h(h) + w1(h, idx_col);
              end loop;
            end if;

            if idx_col = G_N_INPUTS - 1 then
              state <= ST_H_FIRE;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_H_FIRE =>
            leak_div_v := integer(2 ** G_LEAK_SHIFT);

            for h in 0 to G_N_HIDDEN - 1 loop
              mem_v := mem_h(h) - (mem_h(h) / leak_div_v);
              mem_v := mem_v + cur_h(h);

              if mem_v >= G_THRESHOLD_H then
                spike_h(h) <= '1';
                cnt_h(h)   <= cnt_h(h) + 1;
                mem_h(h)   <= mem_v - G_THRESHOLD_H;
              else
                spike_h(h) <= '0';
                mem_h(h)   <= mem_v;
              end if;
            end loop;

            for o in 0 to G_N_OUTPUTS - 1 loop
              cur_o(o) <= 0;
            end loop;

            idx_col <= 0;
            state   <= ST_O_ACCUM;

          when ST_O_ACCUM =>
            if spike_h(idx_col) = '1' then
              for o in 0 to G_N_OUTPUTS - 1 loop
                cur_o(o) <= cur_o(o) + w2(o, idx_col);
              end loop;
            end if;

            if idx_col = G_N_HIDDEN - 1 then
              state <= ST_O_FIRE;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_O_FIRE =>
            leak_div_v := integer(2 ** G_LEAK_SHIFT);

            for o in 0 to G_N_OUTPUTS - 1 loop
              mem_v := mem_o(o) - (mem_o(o) / leak_div_v);
              mem_v := mem_v + cur_o(o);

              if mem_v >= G_THRESHOLD_O then
                spike_o(o) <= '1';
                cnt_o(o)   <= cnt_o(o) + 1;

                if G_OUTPUT_RESET_NONE then
                  mem_o(o) <= mem_v;
                else
                  mem_o(o) <= mem_v - G_THRESHOLD_O;
                end if;
              else
                spike_o(o) <= '0';
                mem_o(o)   <= mem_v;
              end if;
            end loop;

            if step_idx = G_TIMESTEPS - 1 then
              state <= ST_PREP_ERR;
            else
              step_idx <= step_idx + 1;
              state    <= ST_ENCODE;
            end if;

          when ST_PREP_ERR =>
            for o in 0 to G_N_OUTPUTS - 1 loop
              if o = label_reg then
                out_err(o) <= G_TIMESTEPS - cnt_o(o);
              else
                out_err(o) <= -cnt_o(o);
              end if;
            end loop;

            idx_row <= 0;
            idx_col <= 0;

            if train_reg = '1' then
              state <= ST_LEARN_W2;
            else
              state <= ST_ARGMAX;
            end if;

          when ST_LEARN_W2 =>
            upd_v := (G_LR_W2 * out_err(idx_row) * cnt_h(idx_col)) / G_TIMESTEPS;
            upd_v := sat_int(upd_v, -G_MAX_UPD, G_MAX_UPD);
            w2(idx_row, idx_col) <= sat_int(w2(idx_row, idx_col) + upd_v, G_W_MIN, G_W_MAX);

            if idx_col = G_N_HIDDEN - 1 then
              idx_col <= 0;
              if idx_row = G_N_OUTPUTS - 1 then
                idx_row <= 0;
                if G_ENABLE_W1_TRAINING then
                  state <= ST_PREP_HERR;
                else
                  state <= ST_ARGMAX;
                end if;
              else
                idx_row <= idx_row + 1;
              end if;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_PREP_HERR =>
            for h in 0 to G_N_HIDDEN - 1 loop
              sum_v := 0;
              for o in 0 to G_N_OUTPUTS - 1 loop
                sum_v := sum_v + (fb(h, o) * out_err(o));
              end loop;
              hidden_err(h) <= sat_int(sum_v, -G_HERR_CLIP, G_HERR_CLIP);
            end loop;

            idx_row <= 0;
            idx_col <= 0;
            state   <= ST_LEARN_W1;

          when ST_LEARN_W1 =>
            upd_v := (G_LR_W1 * hidden_err(idx_row) * cnt_in(idx_col)) / G_TIMESTEPS;
            upd_v := sat_int(upd_v, -G_MAX_UPD, G_MAX_UPD);
            w1(idx_row, idx_col) <= sat_int(w1(idx_row, idx_col) + upd_v, G_W_MIN, G_W_MAX);

            if idx_col = G_N_INPUTS - 1 then
              idx_col <= 0;
              if idx_row = G_N_HIDDEN - 1 then
                idx_row <= 0;
                state   <= ST_ARGMAX;
              else
                idx_row <= idx_row + 1;
              end if;
            else
              idx_col <= idx_col + 1;
            end if;

          when ST_ARGMAX =>
            pred_v   := argmax(cnt_o);
            pred_reg <= sat_int(pred_v, 0, 15);

            if pred_v = label_reg then
              correct_r <= '1';
            else
              correct_r <= '0';
            end if;

            state <= ST_DONE;

          when ST_DONE =>
            ready_r <= '1';

            if start_i = '0' then
              state <= ST_IDLE;
            end if;

          when others =>
            state <= ST_IDLE;
        end case;
      end if;
    end if;
  end process;
end architecture;
