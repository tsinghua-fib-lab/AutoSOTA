library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.vgg7_pkg.all;

entity vgg7_cifar10_train_top is
  port (
    clk             : in  std_logic;
    rst             : in  std_logic;
    start           : in  std_logic;
    train_mode      : in  std_logic;
    sample_rgb_flat : in  std_logic_vector(C_IN_PIXELS * C_PIX_W - 1 downto 0);
    sample_label    : in  label_t;
    ready           : out std_logic;
    done            : out std_logic;
    pred_label      : out label_t;
    correct         : out std_logic
  );
end entity;

architecture rtl of vgg7_cifar10_train_top is

  type state_t is (
    S_IDLE,
    S_CONV1,
    S_CONV2,
    S_POOL1,
    S_CONV3,
    S_CONV4,
    S_POOL2,
    S_CONV5,
    S_CONV6,
    S_POOL3,
    S_FC,
    S_ERR,
    S_UPD_FC,
    S_UPD_C6,
    S_UPD_C5,
    S_UPD_C4,
    S_UPD_C3,
    S_UPD_C2,
    S_UPD_C1,
    S_DONE
  );
  signal state : state_t := S_IDLE;

  signal fmap0    : act_vec_t(0 to C_FMAP0_LEN-1) := (others => (others => '0'));
  signal fmap1    : act_vec_t(0 to C_FMAP1_LEN-1) := (others => (others => '0'));
  signal fmap2    : act_vec_t(0 to C_FMAP2_LEN-1) := (others => (others => '0'));
  signal pool1    : act_vec_t(0 to C_POOL1_LEN-1) := (others => (others => '0'));
  signal fmap3    : act_vec_t(0 to C_FMAP3_LEN-1) := (others => (others => '0'));
  signal fmap4    : act_vec_t(0 to C_FMAP4_LEN-1) := (others => (others => '0'));
  signal pool2    : act_vec_t(0 to C_POOL2_LEN-1) := (others => (others => '0'));
  signal fmap5    : act_vec_t(0 to C_FMAP5_LEN-1) := (others => (others => '0'));
  signal fmap6    : act_vec_t(0 to C_FMAP6_LEN-1) := (others => (others => '0'));
  signal pool3    : act_vec_t(0 to C_POOL3_LEN-1) := (others => (others => '0'));

  signal w1       : weight_vec_t(0 to C_W1_LEN-1) := init_weight_vec(C_W1_LEN, C_W1_SEED);
  signal w2       : weight_vec_t(0 to C_W2_LEN-1) := init_weight_vec(C_W2_LEN, C_W2_SEED);
  signal w3       : weight_vec_t(0 to C_W3_LEN-1) := init_weight_vec(C_W3_LEN, C_W3_SEED);
  signal w4       : weight_vec_t(0 to C_W4_LEN-1) := init_weight_vec(C_W4_LEN, C_W4_SEED);
  signal w5       : weight_vec_t(0 to C_W5_LEN-1) := init_weight_vec(C_W5_LEN, C_W5_SEED);
  signal w6       : weight_vec_t(0 to C_W6_LEN-1) := init_weight_vec(C_W6_LEN, C_W6_SEED);
  signal w7       : weight_vec_t(0 to C_W7_LEN-1) := init_weight_vec(C_W7_LEN, C_W7_SEED);

  signal scores_reg : score_vec_t(0 to C_CLASSES-1) := (others => (others => '0'));

  signal out_sign : sign_vec_t(0 to C_CLASSES-1) := (others => 0);
  signal teach1   : sign_vec_t(0 to C_C1-1) := (others => 0);
  signal teach2   : sign_vec_t(0 to C_C2-1) := (others => 0);
  signal teach3   : sign_vec_t(0 to C_C3-1) := (others => 0);
  signal teach4   : sign_vec_t(0 to C_C4-1) := (others => 0);
  signal teach5   : sign_vec_t(0 to C_C5-1) := (others => 0);
  signal teach6   : sign_vec_t(0 to C_C6-1) := (others => 0);

  signal label_reg    : label_t := (others => '0');
  signal train_reg    : std_logic := '0';
  signal pred_label_i : label_t := (others => '0');

  constant FB1 : sign_mat_t(0 to C_C1-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C1, C_CLASSES, C_FB1_SEED);
  constant FB2 : sign_mat_t(0 to C_C2-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C2, C_CLASSES, C_FB2_SEED);
  constant FB3 : sign_mat_t(0 to C_C3-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C3, C_CLASSES, C_FB3_SEED);
  constant FB4 : sign_mat_t(0 to C_C4-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C4, C_CLASSES, C_FB4_SEED);
  constant FB5 : sign_mat_t(0 to C_C5-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C5, C_CLASSES, C_FB5_SEED);
  constant FB6 : sign_mat_t(0 to C_C6-1, 0 to C_CLASSES-1) := init_feedback_matrix(C_C6, C_CLASSES, C_FB6_SEED);

  procedure conv3x3_same_relu(
    constant in_h     : in natural;
    constant in_w     : in natural;
    constant in_ch    : in natural;
    constant out_ch   : in natural;
    constant in_fmap  : in act_vec_t;
    constant weights  : in weight_vec_t;
    variable out_fmap : out act_vec_t
  ) is
    variable sum_i : integer;
    variable in_y  : integer;
    variable in_x  : integer;
    variable denom : integer;
  begin
    denom := integer(in_ch) * integer(C_CONV_DIV_PER_IN_CH);
    if denom < 1 then
      denom := 1;
    end if;

    for oc in 0 to out_ch-1 loop
      for y in 0 to in_h-1 loop
        for x in 0 to in_w-1 loop
          sum_i := 0;
          for ic in 0 to in_ch-1 loop
            for ky in 0 to 2 loop
              in_y := integer(y) + integer(ky) - 1;
              if (in_y >= 0) and (in_y < integer(in_h)) then
                for kx in 0 to 2 loop
                  in_x := integer(x) + integer(kx) - 1;
                  if (in_x >= 0) and (in_x < integer(in_w)) then
                    sum_i := sum_i
                      + to_integer(in_fmap(idx3(ic, natural(in_y), natural(in_x), in_h, in_w)))
                      * to_integer(weights(widx4(oc, ic, ky, kx, in_ch, 3, 3)));
                  end if;
                end loop;
              end if;
            end loop;
          end loop;
          out_fmap(idx3(oc, y, x, in_h, in_w)) := relu_act(sum_i / denom);
        end loop;
      end loop;
    end loop;
  end procedure;

  procedure maxpool2x2(
    constant in_ch    : in natural;
    constant in_h     : in natural;
    constant in_w     : in natural;
    constant in_fmap  : in act_vec_t;
    variable out_fmap : out act_vec_t
  ) is
    variable best_i : integer;
    variable temp_i : integer;
    constant out_h  : natural := in_h / 2;
    constant out_w  : natural := in_w / 2;
  begin
    for c in 0 to in_ch-1 loop
      for y in 0 to out_h-1 loop
        for x in 0 to out_w-1 loop
          best_i := to_integer(in_fmap(idx3(c, 2*y, 2*x, in_h, in_w)));
          temp_i := to_integer(in_fmap(idx3(c, 2*y, 2*x + 1, in_h, in_w)));
          if temp_i > best_i then
            best_i := temp_i;
          end if;
          temp_i := to_integer(in_fmap(idx3(c, 2*y + 1, 2*x, in_h, in_w)));
          if temp_i > best_i then
            best_i := temp_i;
          end if;
          temp_i := to_integer(in_fmap(idx3(c, 2*y + 1, 2*x + 1, in_h, in_w)));
          if temp_i > best_i then
            best_i := temp_i;
          end if;
          out_fmap(idx3(c, y, x, out_h, out_w)) := sat_act_signed(best_i);
        end loop;
      end loop;
    end loop;
  end procedure;

  procedure fc_forward(
    constant n_in     : in natural;
    constant n_out    : in natural;
    constant in_vec   : in act_vec_t;
    constant weights  : in weight_vec_t;
    variable scores   : out score_vec_t
  ) is
    variable sum_i : integer;
    variable denom : integer;
  begin
    denom := integer(n_in);
    if denom < 1 then
      denom := 1;
    end if;

    for o in 0 to n_out-1 loop
      sum_i := 0;
      for i in 0 to n_in-1 loop
        sum_i := sum_i + to_integer(in_vec(i)) * to_integer(weights(fcidx(o, i, n_in)));
      end loop;
      scores(o) := sat_score(sum_i / denom);
    end loop;
  end procedure;

  procedure compute_output_signs(
    constant scores      : in score_vec_t;
    constant target_lab  : in label_t;
    variable out_sgn     : out sign_vec_t
  ) is
    variable err_i : integer;
  begin
    for k in 0 to C_CLASSES-1 loop
      if k = to_integer(target_lab) then
        err_i := C_TARGET_SCORE - to_integer(scores(k));
      else
        err_i := -to_integer(scores(k));
      end if;
      out_sgn(k) := sign_i(err_i);
    end loop;
  end procedure;

  procedure apply_feedback(
    constant fb       : in sign_mat_t;
    constant out_sgn  : in sign_vec_t;
    variable teach    : out sign_vec_t
  ) is
    variable sum_i : integer;
  begin
    for row in teach'range loop
      sum_i := 0;
      for k in out_sgn'range loop
        sum_i := sum_i + fb(row, k) * out_sgn(k);
      end loop;
      teach(row) := sign_i(sum_i);
    end loop;
  end procedure;

  procedure channel_signs(
    constant n_ch     : in natural;
    constant h        : in natural;
    constant w        : in natural;
    constant fmap     : in act_vec_t;
    variable signs    : out sign_vec_t
  ) is
    variable sum_i : integer;
  begin
    for c in 0 to n_ch-1 loop
      sum_i := 0;
      for y in 0 to h-1 loop
        for x in 0 to w-1 loop
          sum_i := sum_i + to_integer(fmap(idx3(c, y, x, h, w)));
        end loop;
      end loop;
      signs(c) := sign_i(sum_i);
    end loop;
  end procedure;

  procedure fc_sign_update(
    constant n_in      : in natural;
    constant n_out     : in natural;
    constant in_vec    : in act_vec_t;
    constant out_sgn   : in sign_vec_t;
    variable weights   : inout weight_vec_t
  ) is
    variable delta : integer;
    variable idx   : natural;
  begin
    for o in 0 to n_out-1 loop
      for i in 0 to n_in-1 loop
        idx := fcidx(o, i, n_in);
        delta := C_LR_FC * sign_i(to_integer(in_vec(i))) * out_sgn(o);
        if delta /= 0 then
          weights(idx) := sat_weight_add(weights(idx), delta);
        end if;
      end loop;
    end loop;
  end procedure;

  procedure conv_channel_update(
    constant in_ch      : in natural;
    constant in_h       : in natural;
    constant in_w       : in natural;
    constant out_ch     : in natural;
    constant out_h      : in natural;
    constant out_w      : in natural;
    constant pre_fmap   : in act_vec_t;
    constant post_fmap  : in act_vec_t;
    constant teach      : in sign_vec_t;
    variable weights    : inout weight_vec_t
  ) is
    variable pre_sgn  : sign_vec_t(0 to in_ch-1);
    variable post_sgn : sign_vec_t(0 to out_ch-1);
    variable delta    : integer;
    variable idx      : natural;
  begin
    channel_signs(in_ch, in_h, in_w, pre_fmap, pre_sgn);
    channel_signs(out_ch, out_h, out_w, post_fmap, post_sgn);

    for oc in 0 to out_ch-1 loop
      for ic in 0 to in_ch-1 loop
        delta := C_LR_CONV * pre_sgn(ic) * post_sgn(oc) * teach(oc);
        if delta /= 0 then
          for ky in 0 to 2 loop
            for kx in 0 to 2 loop
              idx := widx4(oc, ic, ky, kx, in_ch, 3, 3);
              weights(idx) := sat_weight_add(weights(idx), delta);
            end loop;
          end loop;
        end if;
      end loop;
    end loop;
  end procedure;

begin

  ready        <= '1' when state = S_IDLE else '0';
  done         <= '1' when state = S_DONE else '0';
  pred_label_i <= argmax_scores(scores_reg);
  pred_label   <= pred_label_i;
  correct      <= '1' when pred_label_i = label_reg else '0';

  process(clk)
    variable pix_v       : pixel_vec_t(0 to C_IN_PIXELS-1);
    variable v_f0        : act_vec_t(0 to C_FMAP0_LEN-1);
    variable v_f1        : act_vec_t(0 to C_FMAP1_LEN-1);
    variable v_f2        : act_vec_t(0 to C_FMAP2_LEN-1);
    variable v_p1        : act_vec_t(0 to C_POOL1_LEN-1);
    variable v_f3        : act_vec_t(0 to C_FMAP3_LEN-1);
    variable v_f4        : act_vec_t(0 to C_FMAP4_LEN-1);
    variable v_p2        : act_vec_t(0 to C_POOL2_LEN-1);
    variable v_f5        : act_vec_t(0 to C_FMAP5_LEN-1);
    variable v_f6        : act_vec_t(0 to C_FMAP6_LEN-1);
    variable v_p3        : act_vec_t(0 to C_POOL3_LEN-1);

    variable v_scores    : score_vec_t(0 to C_CLASSES-1);

    variable v_w1        : weight_vec_t(0 to C_W1_LEN-1);
    variable v_w2        : weight_vec_t(0 to C_W2_LEN-1);
    variable v_w3        : weight_vec_t(0 to C_W3_LEN-1);
    variable v_w4        : weight_vec_t(0 to C_W4_LEN-1);
    variable v_w5        : weight_vec_t(0 to C_W5_LEN-1);
    variable v_w6        : weight_vec_t(0 to C_W6_LEN-1);
    variable v_w7        : weight_vec_t(0 to C_W7_LEN-1);

    variable v_out_sign  : sign_vec_t(0 to C_CLASSES-1);
    variable v_teach1    : sign_vec_t(0 to C_C1-1);
    variable v_teach2    : sign_vec_t(0 to C_C2-1);
    variable v_teach3    : sign_vec_t(0 to C_C3-1);
    variable v_teach4    : sign_vec_t(0 to C_C4-1);
    variable v_teach5    : sign_vec_t(0 to C_C5-1);
    variable v_teach6    : sign_vec_t(0 to C_C6-1);
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state      <= S_IDLE;
        label_reg  <= (others => '0');
        train_reg  <= '0';

        fmap0      <= (others => (others => '0'));
        fmap1      <= (others => (others => '0'));
        fmap2      <= (others => (others => '0'));
        pool1      <= (others => (others => '0'));
        fmap3      <= (others => (others => '0'));
        fmap4      <= (others => (others => '0'));
        pool2      <= (others => (others => '0'));
        fmap5      <= (others => (others => '0'));
        fmap6      <= (others => (others => '0'));
        pool3      <= (others => (others => '0'));
        scores_reg <= (others => (others => '0'));

        out_sign   <= (others => 0);
        teach1     <= (others => 0);
        teach2     <= (others => 0);
        teach3     <= (others => 0);
        teach4     <= (others => 0);
        teach5     <= (others => 0);
        teach6     <= (others => 0);

        w1         <= init_weight_vec(C_W1_LEN, C_W1_SEED);
        w2         <= init_weight_vec(C_W2_LEN, C_W2_SEED);
        w3         <= init_weight_vec(C_W3_LEN, C_W3_SEED);
        w4         <= init_weight_vec(C_W4_LEN, C_W4_SEED);
        w5         <= init_weight_vec(C_W5_LEN, C_W5_SEED);
        w6         <= init_weight_vec(C_W6_LEN, C_W6_SEED);
        w7         <= init_weight_vec(C_W7_LEN, C_W7_SEED);

      else
        case state is
          when S_IDLE =>
            if start = '1' then
              pix_v := unpack_pixels(sample_rgb_flat);
              for i in 0 to C_IN_PIXELS-1 loop
                v_f0(i) := pixel_to_act(pix_v(i));
              end loop;
              fmap0     <= v_f0;
              label_reg <= sample_label;
              train_reg <= train_mode;
              state     <= S_CONV1;
            end if;

          when S_CONV1 =>
            conv3x3_same_relu(C_IMG_H, C_IMG_W, C_IMG_CH, C_C1, fmap0, w1, v_f1);
            fmap1 <= v_f1;
            state <= S_CONV2;

          when S_CONV2 =>
            conv3x3_same_relu(C_F1_H, C_F1_W, C_C1, C_C2, fmap1, w2, v_f2);
            fmap2 <= v_f2;
            state <= S_POOL1;

          when S_POOL1 =>
            maxpool2x2(C_C2, C_F1_H, C_F1_W, fmap2, v_p1);
            pool1 <= v_p1;
            state <= S_CONV3;

          when S_CONV3 =>
            conv3x3_same_relu(C_P1_H, C_P1_W, C_C2, C_C3, pool1, w3, v_f3);
            fmap3 <= v_f3;
            state <= S_CONV4;

          when S_CONV4 =>
            conv3x3_same_relu(C_F3_H, C_F3_W, C_C3, C_C4, fmap3, w4, v_f4);
            fmap4 <= v_f4;
            state <= S_POOL2;

          when S_POOL2 =>
            maxpool2x2(C_C4, C_F3_H, C_F3_W, fmap4, v_p2);
            pool2 <= v_p2;
            state <= S_CONV5;

          when S_CONV5 =>
            conv3x3_same_relu(C_P2_H, C_P2_W, C_C4, C_C5, pool2, w5, v_f5);
            fmap5 <= v_f5;
            state <= S_CONV6;

          when S_CONV6 =>
            conv3x3_same_relu(C_F5_H, C_F5_W, C_C5, C_C6, fmap5, w6, v_f6);
            fmap6 <= v_f6;
            state <= S_POOL3;

          when S_POOL3 =>
            maxpool2x2(C_C6, C_F5_H, C_F5_W, fmap6, v_p3);
            pool3 <= v_p3;
            state <= S_FC;

          when S_FC =>
            fc_forward(C_FC_IN, C_CLASSES, pool3, w7, v_scores);
            scores_reg <= v_scores;
            if train_reg = '1' then
              state <= S_ERR;
            else
              state <= S_DONE;
            end if;

          when S_ERR =>
            compute_output_signs(scores_reg, label_reg, v_out_sign);
            apply_feedback(FB1, v_out_sign, v_teach1);
            apply_feedback(FB2, v_out_sign, v_teach2);
            apply_feedback(FB3, v_out_sign, v_teach3);
            apply_feedback(FB4, v_out_sign, v_teach4);
            apply_feedback(FB5, v_out_sign, v_teach5);
            apply_feedback(FB6, v_out_sign, v_teach6);

            out_sign <= v_out_sign;
            teach1   <= v_teach1;
            teach2   <= v_teach2;
            teach3   <= v_teach3;
            teach4   <= v_teach4;
            teach5   <= v_teach5;
            teach6   <= v_teach6;

            state    <= S_UPD_FC;

          when S_UPD_FC =>
            v_w7 := w7;
            fc_sign_update(C_FC_IN, C_CLASSES, pool3, out_sign, v_w7);
            w7    <= v_w7;
            if C_USE_ALL_LAYER_UPDATES then
              state <= S_UPD_C6;
            else
              state <= S_DONE;
            end if;

          when S_UPD_C6 =>
            v_w6 := w6;
            conv_channel_update(C_C5, C_F5_H, C_F5_W, C_C6, C_F5_H, C_F5_W, fmap5, fmap6, teach6, v_w6);
            w6    <= v_w6;
            state <= S_UPD_C5;

          when S_UPD_C5 =>
            v_w5 := w5;
            conv_channel_update(C_C4, C_P2_H, C_P2_W, C_C5, C_F5_H, C_F5_W, pool2, fmap5, teach5, v_w5);
            w5    <= v_w5;
            state <= S_UPD_C4;

          when S_UPD_C4 =>
            v_w4 := w4;
            conv_channel_update(C_C3, C_F3_H, C_F3_W, C_C4, C_F3_H, C_F3_W, fmap3, fmap4, teach4, v_w4);
            w4    <= v_w4;
            state <= S_UPD_C3;

          when S_UPD_C3 =>
            v_w3 := w3;
            conv_channel_update(C_C2, C_P1_H, C_P1_W, C_C3, C_F3_H, C_F3_W, pool1, fmap3, teach3, v_w3);
            w3    <= v_w3;
            state <= S_UPD_C2;

          when S_UPD_C2 =>
            v_w2 := w2;
            conv_channel_update(C_C1, C_F1_H, C_F1_W, C_C2, C_F1_H, C_F1_W, fmap1, fmap2, teach2, v_w2);
            w2    <= v_w2;
            state <= S_UPD_C1;

          when S_UPD_C1 =>
            v_w1 := w1;
            conv_channel_update(C_IMG_CH, C_IMG_H, C_IMG_W, C_C1, C_F1_H, C_F1_W, fmap0, fmap1, teach1, v_w1);
            w1    <= v_w1;
            state <= S_DONE;

          when S_DONE =>
            state <= S_IDLE;

        end case;
      end if;
    end if;
  end process;

end architecture;
