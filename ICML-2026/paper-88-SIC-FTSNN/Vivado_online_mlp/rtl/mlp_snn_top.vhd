library ieee;
use ieee.std_logic_1164.all;

entity mlp_snn_top is
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

architecture rtl of mlp_snn_top is
begin
  u_core : entity work.mlp_snn_core
    generic map(
      G_N_INPUTS           => G_N_INPUTS,
      G_N_HIDDEN           => G_N_HIDDEN,
      G_N_OUTPUTS          => G_N_OUTPUTS,
      G_TIMESTEPS          => G_TIMESTEPS,
      G_LEAK_SHIFT         => G_LEAK_SHIFT,
      G_THRESHOLD_H        => G_THRESHOLD_H,
      G_THRESHOLD_O        => G_THRESHOLD_O,
      G_W_MIN              => G_W_MIN,
      G_W_MAX              => G_W_MAX,
      G_INIT_SPREAD        => G_INIT_SPREAD,
      G_MAX_UPD            => G_MAX_UPD,
      G_LR_W1              => G_LR_W1,
      G_LR_W2              => G_LR_W2,
      G_HERR_CLIP          => G_HERR_CLIP,
      G_ENABLE_W1_TRAINING => G_ENABLE_W1_TRAINING,
      G_OUTPUT_RESET_NONE  => G_OUTPUT_RESET_NONE
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
end architecture;
