# Vivado TCL script for behavioral simulation
# Usage:
#   vivado -mode batch -source run_sim.tcl
#   vivado -mode batch -source run_sim.tcl -tclargs 128 64 1
#
# Positional args:
#   0: number of training samples
#   1: number of test samples
#   2: number of epochs

set num_train 128
set num_test 64
set epochs 1

if {$argc >= 1} {
  set num_train [lindex $argv 0]
}
if {$argc >= 2} {
  set num_test [lindex $argv 1]
}
if {$argc >= 3} {
  set epochs [lindex $argv 2]
}

set root_dir [file normalize [file join [pwd] ".."]]
set train_file [file normalize [file join $root_dir data cifar10_train.txt]]
set test_file  [file normalize [file join $root_dir data cifar10_test.txt]]

create_project vgg7_cifar10_sim ./vivado_sim -part xc7z020clg400-1 -force
set_property target_language VHDL [current_project]

read_vhdl -vhdl2008 ../rtl/vgg7_pkg.vhd
read_vhdl -vhdl2008 ../rtl/vgg7_cifar10_train_top.vhd
read_vhdl -vhdl2008 ../tb/tb_vgg7_cifar10.vhd

set_property top tb_vgg7_cifar10 [current_fileset]
set_property generic "G_TRAIN_FILE=$train_file G_TEST_FILE=$test_file G_NUM_TRAIN=$num_train G_NUM_TEST=$num_test G_EPOCHS=$epochs" [current_fileset]

launch_simulation
run all
quit
