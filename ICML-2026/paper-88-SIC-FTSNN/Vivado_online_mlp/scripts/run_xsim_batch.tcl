set dataset "mnist"
set part_name "xc7z020clg400-1"
set train_samples 128
set test_samples 32

if {$argc >= 1} { set dataset [string tolower [lindex $argv 0]] }
if {$argc >= 2} { set part_name [lindex $argv 1] }
if {$argc >= 3} { set train_samples [lindex $argv 2] }
if {$argc >= 4} { set test_samples [lindex $argv 3] }

set root_dir [file normalize [file join [file dirname [info script]] ".."]]
set proj_dir [file join $root_dir "vivado_sim"]

if {$dataset eq "fmnist"} {
  set train_img [file normalize [file join $root_dir "data" "fmnist_train_images.txt"]]
  set train_lbl [file normalize [file join $root_dir "data" "fmnist_train_labels.txt"]]
  set test_img  [file normalize [file join $root_dir "data" "fmnist_test_images.txt"]]
  set test_lbl  [file normalize [file join $root_dir "data" "fmnist_test_labels.txt"]]
} else {
  set dataset "mnist"
  set train_img [file normalize [file join $root_dir "data" "mnist_train_images.txt"]]
  set train_lbl [file normalize [file join $root_dir "data" "mnist_train_labels.txt"]]
  set test_img  [file normalize [file join $root_dir "data" "mnist_test_images.txt"]]
  set test_lbl  [file normalize [file join $root_dir "data" "mnist_test_labels.txt"]]
}

create_project spikerplus_onchip_mlp_snn_sim $proj_dir -part $part_name -force

add_files -fileset sources_1 [list \
  [file join $root_dir "rtl" "snn_pkg.vhd"] \
  [file join $root_dir "rtl" "mlp_snn_core.vhd"] \
  [file join $root_dir "rtl" "mlp_snn_top.vhd"] \
]

add_files -fileset sim_1 [list \
  [file join $root_dir "tb" "tb_mlp_snn.vhd"] \
]

foreach fs {sources_1 sim_1} {
  foreach f [get_files -of_objects [get_filesets $fs]] {
    set_property file_type {VHDL 2008} [get_files $f]
  }
}

set_property top tb_mlp_snn [get_filesets sim_1]
set_property generic [format {G_DATASET_NAME="%s" G_TRAIN_IMAGE_FILE="%s" G_TRAIN_LABEL_FILE="%s" G_TEST_IMAGE_FILE="%s" G_TEST_LABEL_FILE="%s" G_TRAIN_SAMPLES=%d G_TEST_SAMPLES=%d} \
  $dataset $train_img $train_lbl $test_img $test_lbl $train_samples $test_samples] [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

launch_simulation -mode batch
run all
close_sim
close_project
puts "Batch simulation complete for dataset=$dataset"
