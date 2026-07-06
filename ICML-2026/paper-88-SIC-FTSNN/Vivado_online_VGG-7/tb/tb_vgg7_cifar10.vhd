library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

library work;
use work.vgg7_pkg.all;

entity tb_vgg7_cifar10 is
  generic (
    G_TRAIN_FILE : string  := "../data/cifar10_train.txt";
    G_TEST_FILE  : string  := "../data/cifar10_test.txt";
    G_NUM_TRAIN  : natural := 64;
    G_NUM_TEST   : natural := 64;
    G_EPOCHS     : natural := 1;
    G_CLK_PERIOD : time    := 10 ns
  );
end entity;

architecture sim of tb_vgg7_cifar10 is

  signal clk             : std_logic := '0';
  signal rst             : std_logic := '1';
  signal start           : std_logic := '0';
  signal train_mode      : std_logic := '0';
  signal ready           : std_logic;
  signal done            : std_logic;
  signal pred_label      : label_t;
  signal correct         : std_logic;
  signal sample_rgb_flat : std_logic_vector(C_IN_PIXELS * C_PIX_W - 1 downto 0) := (others => '0');
  signal sample_label    : label_t := (others => '0');

  procedure read_one_sample(
    file f        : text;
    variable ok   : out boolean;
    variable y    : out label_t;
    variable pix  : out pixel_vec_t(0 to C_IN_PIXELS-1)
  ) is
    variable l     : line;
    variable value : integer;
  begin
    if endfile(f) then
      ok := false;
      y  := (others => '0');
      for i in 0 to C_IN_PIXELS-1 loop
        pix(i) := (others => '0');
      end loop;
      return;
    end if;

    readline(f, l);
    read(l, value);
    y := to_unsigned(value, C_LABEL_W);

    for i in 0 to C_IN_PIXELS-1 loop
      read(l, value);
      if value < 0 then
        value := 0;
      elsif value > 255 then
        value := 255;
      end if;
      pix(i) := to_unsigned(value, C_PIX_W);
    end loop;

    ok := true;
  end procedure;

begin

  clk <= not clk after G_CLK_PERIOD / 2;

  dut : entity work.vgg7_cifar10_train_top
    port map (
      clk             => clk,
      rst             => rst,
      start           => start,
      train_mode      => train_mode,
      sample_rgb_flat => sample_rgb_flat,
      sample_label    => sample_label,
      ready           => ready,
      done            => done,
      pred_label      => pred_label,
      correct         => correct
    );

  stim : process
    file train_f : text;
    file test_f  : text;

    variable ok          : boolean;
    variable pix_v       : pixel_vec_t(0 to C_IN_PIXELS-1);
    variable lab_v       : label_t;
    variable test_hits   : integer := 0;
    variable n_test_seen : integer := 0;
  begin
    report "Reset asserted.";
    wait for 100 ns;
    rst <= '0';
    wait for 50 ns;

    file_open(train_f, G_TRAIN_FILE, read_mode);
    report "Opened training file: " & G_TRAIN_FILE;

    for epoch in 1 to G_EPOCHS loop
      for s in 1 to G_NUM_TRAIN loop
        read_one_sample(train_f, ok, lab_v, pix_v);
        exit when not ok;

        wait until ready = '1';
        sample_rgb_flat <= pack_pixels(pix_v);
        sample_label    <= lab_v;
        train_mode      <= '1';
        start           <= '1';
        wait until rising_edge(clk);
        start           <= '0';

        wait until done = '1';
        wait until rising_edge(clk);
      end loop;

      if epoch < G_EPOCHS then
        file_close(train_f);
        file_open(train_f, G_TRAIN_FILE, read_mode);
      end if;

      report "Finished training epoch " & integer'image(epoch) & ".";
    end loop;

    file_close(train_f);

    file_open(test_f, G_TEST_FILE, read_mode);
    report "Opened test file: " & G_TEST_FILE;

    for s in 1 to G_NUM_TEST loop
      read_one_sample(test_f, ok, lab_v, pix_v);
      exit when not ok;

      wait until ready = '1';
      sample_rgb_flat <= pack_pixels(pix_v);
      sample_label    <= lab_v;
      train_mode      <= '0';
      start           <= '1';
      wait until rising_edge(clk);
      start           <= '0';

      wait until done = '1';

      if correct = '1' then
        test_hits := test_hits + 1;
      end if;
      n_test_seen := n_test_seen + 1;

      report "Test sample " & integer'image(n_test_seen)
        & " label=" & integer'image(to_integer(lab_v))
        & " pred=" & integer'image(to_integer(pred_label));

      wait until rising_edge(clk);
    end loop;

    file_close(test_f);

    report "Test accuracy = "
      & integer'image(test_hits) & "/"
      & integer'image(n_test_seen);

    report "Simulation finished.";
    wait;
  end process;

end architecture;
