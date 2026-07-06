library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_mlp_snn is
  generic(
    G_DATASET_NAME      : string  := "mnist";
    G_TRAIN_IMAGE_FILE  : string  := "data/mnist_train_images.txt";
    G_TRAIN_LABEL_FILE  : string  := "data/mnist_train_labels.txt";
    G_TEST_IMAGE_FILE   : string  := "data/mnist_test_images.txt";
    G_TEST_LABEL_FILE   : string  := "data/mnist_test_labels.txt";
    G_N_INPUTS          : positive := 784;
    G_N_HIDDEN          : positive := 64;
    G_N_OUTPUTS         : positive := 10;
    G_TIMESTEPS         : positive := 10;
    G_TRAIN_SAMPLES     : positive := 128;
    G_TEST_SAMPLES      : positive := 32;
    G_ENABLE_W1_TRAINING: boolean  := true
  );
end entity;

architecture sim of tb_mlp_snn is
  constant C_CLK_PERIOD : time := 10 ns;

  signal clk          : std_logic := '0';
  signal rst          : std_logic := '1';
  signal start_i      : std_logic := '0';
  signal train_i      : std_logic := '0';
  signal image_i      : std_logic_vector((G_N_INPUTS * 8) - 1 downto 0) := (others => '0');
  signal label_i      : std_logic_vector(3 downto 0) := (others => '0');
  signal ready_o      : std_logic;
  signal pred_label_o : std_logic_vector(3 downto 0);
  signal correct_o    : std_logic;

  procedure read_image(
    file f          : text;
    variable img_v  : out std_logic_vector
  ) is
    variable l      : line;
    variable pix    : integer;
  begin
    readline(f, l);
    for i in 0 to G_N_INPUTS - 1 loop
      read(l, pix);
      img_v((8 * i) + 7 downto 8 * i) := std_logic_vector(to_unsigned(pix, 8));
    end loop;
  end procedure;

  procedure read_label(
    file f          : text;
    variable lbl_v  : out integer
  ) is
    variable l      : line;
  begin
    readline(f, l);
    read(l, lbl_v);
  end procedure;
begin
  clk <= not clk after C_CLK_PERIOD / 2;

  uut : entity work.mlp_snn_top
    generic map(
      G_N_INPUTS           => G_N_INPUTS,
      G_N_HIDDEN           => G_N_HIDDEN,
      G_N_OUTPUTS          => G_N_OUTPUTS,
      G_TIMESTEPS          => G_TIMESTEPS,
      G_LEAK_SHIFT         => 4,
      G_THRESHOLD_H        => 16,
      G_THRESHOLD_O        => 8,
      G_W_MIN              => -7,
      G_W_MAX              => 7,
      G_INIT_SPREAD        => 1,
      G_MAX_UPD            => 2,
      G_LR_W1              => 1,
      G_LR_W2              => 1,
      G_HERR_CLIP          => 8,
      G_ENABLE_W1_TRAINING => G_ENABLE_W1_TRAINING,
      G_OUTPUT_RESET_NONE  => true
    )
    port map(
      clk          => clk,
      rst          => rst,
      start_i      => start_i,
      train_i      => train_i,
      image_i      => image_i,
      label_i      => label_i,
      ready_o      => ready_o,
      pred_label_o => pred_label_o,
      correct_o    => correct_o
    );

  stimulus : process
    file train_img_f : text open read_mode is G_TRAIN_IMAGE_FILE;
    file train_lbl_f : text open read_mode is G_TRAIN_LABEL_FILE;
    file test_img_f  : text open read_mode is G_TEST_IMAGE_FILE;
    file test_lbl_f  : text open read_mode is G_TEST_LABEL_FILE;

    variable img_v        : std_logic_vector((G_N_INPUTS * 8) - 1 downto 0);
    variable lbl_v        : integer;
    variable train_seen   : integer := 0;
    variable train_ok     : integer := 0;
    variable test_seen    : integer := 0;
    variable test_ok      : integer := 0;
  begin
    report "TB start. Dataset = " & G_DATASET_NAME severity note;

    rst     <= '1';
    start_i <= '0';
    train_i <= '0';
    wait for 20 * C_CLK_PERIOD;
    rst <= '0';

    wait until ready_o = '1';
    wait until rising_edge(clk);

    report "Core initialized. Beginning online training." severity note;

    for n in 0 to G_TRAIN_SAMPLES - 1 loop
      exit when endfile(train_img_f) or endfile(train_lbl_f);

      read_image(train_img_f, img_v);
      read_label(train_lbl_f, lbl_v);

      image_i <= img_v;
      label_i <= std_logic_vector(to_unsigned(lbl_v, 4));
      train_i <= '1';

      wait until rising_edge(clk);
      start_i <= '1';
      wait until rising_edge(clk);
      start_i <= '0';

      wait until ready_o = '0';
      wait until ready_o = '1';

      train_seen := train_seen + 1;
      if correct_o = '1' then
        train_ok := train_ok + 1;
      end if;

      if (train_seen mod 32) = 0 then
        report "Training samples processed = " & integer'image(train_seen) severity note;
      end if;

      wait until rising_edge(clk);
    end loop;

    report "Training phase done. Correct predictions during train passes = " &
           integer'image(train_ok) & "/" & integer'image(train_seen) severity note;

    report "Beginning test/inference pass." severity note;

    for n in 0 to G_TEST_SAMPLES - 1 loop
      exit when endfile(test_img_f) or endfile(test_lbl_f);

      read_image(test_img_f, img_v);
      read_label(test_lbl_f, lbl_v);

      image_i <= img_v;
      label_i <= std_logic_vector(to_unsigned(lbl_v, 4));
      train_i <= '0';

      wait until rising_edge(clk);
      start_i <= '1';
      wait until rising_edge(clk);
      start_i <= '0';

      wait until ready_o = '0';
      wait until ready_o = '1';

      test_seen := test_seen + 1;
      if correct_o = '1' then
        test_ok := test_ok + 1;
      end if;

      wait until rising_edge(clk);
    end loop;

    report "Test phase done. Correct predictions = " &
           integer'image(test_ok) & "/" & integer'image(test_seen) severity note;

    report "Last predicted label = " &
           integer'image(to_integer(unsigned(pred_label_o))) severity note;

    wait;
  end process;
end architecture;
