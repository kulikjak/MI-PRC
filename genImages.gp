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
     "results/fw_cpu_blocked.out" using 1:(($2+$3+$4)/3) with line title "CPU blocked", \
     "results/fw_openacc_blocked.out" using 1:(($2+$4+$6)/3) with line title "GPU without data copy", \
     "results/fw_openacc_blocked.out" using 1:(($3+$5+$7)/3) with line title "GPU with data copy"

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

set output "results/FW_Cuda.pdf"

set title "Floyd Warshall cuda kernels comparison"
set xlabel "order of matrix"
set ylabel "time (s)"

set xtics mirror
set key left top

plot "results/kernels/cuda_fw_0.out" using 1:(($3+$5+$7)/3) with line title "Kernel 0", \
     "results/kernels/cuda_fw_1.out" using 1:(($3+$5+$7)/3) with line title "Kernel 1", \
     "results/kernels/cuda_fw_2.out" using 1:(($3+$5+$7)/3) with line title "Kernel 2", \
     "results/kernels/cuda_fw_3.out" using 1:(($3+$5+$7)/3) with line title "Kernel 3"

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

set output "results/FW_Cuda2.pdf"

set title "Floyd Warshall cuda kernels comparison"
set xlabel "order of matrix"
set ylabel "time (s)"

set xtics mirror
set key left top

plot "results/kernels/cuda_fw_0.out" using 1:(($2+$4+$6)/3) with line title "Kernel 0 without data copy", \
     "results/kernels/cuda_fw_0.out" using 1:(($3+$5+$7)/3) with line title "Kernel 0 with data copy", \
     "results/kernels/cuda_fw_2.out" using 1:(($2+$4+$6)/3) with line title "Kernel 2 without data copy", \
     "results/kernels/cuda_fw_2.out" using 1:(($3+$5+$7)/3) with line title "Kernel 2 with data copy"

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

set output "results/FW_Cuda3.pdf"

set title "Floyd Warshall comparison of Cuda and OpenACC"
set xlabel "order of matrix"
set ylabel "time (s)"

set xtics mirror
set key left top

plot "results/kernels/cuda_fw_2.out" using 1:(($3+$5+$7)/3) with line title "Cuda kernels", \
     "results/fw_openacc_blocked.out" using 1:(($3+$5+$7)/3) with line title "OpenAcc kernels"

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

set output "results/Comparison.pdf"

set title "Floyd Warshall and Dijkstra algorithm comparison"
set xlabel "order of matrix"
set ylabel "time (s)"

set xtics mirror
set key left top

plot "results/kernels/cuda_fw_2.out" using 1:(($3+$5+$7)/3) with line title "Floyd Warshall", \
     "results/cuda_dijkstra.out" using 1:(($3+$5+$7)/3) with line title "Dijkstra"
