# Vivado TCL script for synthesis
# Usage:
#   vivado -mode batch -source run_synth.tcl

create_project vgg7_cifar10_synth ./vivado_synth -part xc7z020clg400-1 -force
set_property target_language VHDL [current_project]

read_vhdl -vhdl2008 ../rtl/vgg7_pkg.vhd
read_vhdl -vhdl2008 ../rtl/vgg7_cifar10_train_top.vhd

set_property top vgg7_cifar10_train_top [current_fileset]

synth_design -top vgg7_cifar10_train_top -part xc7z020clg400-1
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
quit
