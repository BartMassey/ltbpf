# Plot the CSV produced by examples/vehicle.rs using gnuplot.
#
# Usage:
#     cargo run --release --example vehicle > out.csv
#     gnuplot -c examples/plot.gp out.csv               # window
#     gnuplot -c examples/plot.gp out.csv vehicle.png   # file
#
# Three stacked panels: 2D tracks (truth vs estimate), tracking error
# over time, and effective sample size over time.

if (ARGC < 1) {
    print "usage: gnuplot -c plot.gp <csv> [outfile.png]"
    exit 1
}

csv = ARG1

if (ARGC >= 2) {
    set terminal pngcairo size 900,1100 enhanced font ",10"
    set output ARG2
} else {
    set terminal qt size 900,1100 enhanced font ",10"
}

set datafile separator ","
set datafile columnheaders     # use the header line for column names
set grid
set key top left

set multiplot layout 3,1 margins 0.10,0.95,0.07,0.97 spacing 0,0.08

# Panel 1: 2D tracks.
set title "truth vs estimated track"
set xlabel "x"
set ylabel "y"
set size ratio -1               # equal x/y scale
plot \
    csv using "truth_x":"truth_y" with lines lw 1.5 title "truth", \
    csv using "est_x":"est_y"     with lines lw 1.0 title "estimate"
set size noratio

# Panel 2: tracking error.
set title "tracking error (m)"
set xlabel "step"
set ylabel "|truth - estimate|"
unset key
plot csv using "step":"err" with lines lw 0.9 lc rgb "dark-red"

# Panel 3: effective sample size.
set title "effective sample size"
set xlabel "step"
set ylabel "ESS"
plot csv using "step":"ess" with lines lw 0.9 lc rgb "dark-blue"

unset multiplot

if (ARGC >= 2) {
    print sprintf("wrote %s", ARG2)
}
