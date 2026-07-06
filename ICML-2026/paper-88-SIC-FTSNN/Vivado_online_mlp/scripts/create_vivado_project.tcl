set proj_name "spikerplus_onchip_mlp_snn"
set part_name "xc7z020clg400-1"

if {$argc >= 1} {
  set part_name [lindex $argv 0]
}

set root_dir [file normalize [file join [file dirname [info script]] ".."]]
set proj_dir [file join $root_dir "vivado_proj"]

create_project $proj_name $proj_dir -part $part_name -force

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

set_property top mlp_snn_top [get_filesets sources_1]
set_property top tb_mlp_snn  [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "Project created at $proj_dir"
