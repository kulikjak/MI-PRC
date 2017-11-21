reset
set encoding utf8
set terminal pdfcairo size 5, 3.8

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

set output "results/Floyd_Warshall.pdf"

set title "Floyd Warshall algorithm"
set xlabel "order of matrix"
set ylabel "time (s)"

set xtics mirror
set key left top

plot "results/fw_cpu.out" using 1:(($2+$3+$4)/3) with line title "CPU", \
     "results/fw_openacc.out" using 1:(($2+$4+$6)/3) with line title "GPU without data copy", \
     "results/fw_openacc.out" using 1:(($3+$5+$7)/3) with line title "GPU with data copy"
