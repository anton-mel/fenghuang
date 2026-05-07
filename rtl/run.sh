#!/bin/sh
iverilog -g2012 -o tb_fenghuang_tab.vvp ../tb/tb_fenghuang_tab.v *.v
vvp tb_fenghuang_tab.vvp
