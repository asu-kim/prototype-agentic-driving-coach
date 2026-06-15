set datafile separator comma
set terminal pngcairo size 1800,1050 enhanced font "Sans,12"
set output "src/HardwareIntegration/environment_timeline.png"

csv = "src/HardwareIntegration/environment.csv"
dt = 0.1

set multiplot layout 2,1 title "Environment Scenario Timeline (100 ms per CSV row)" font ",18"

set xrange [0:85]
set yrange [-0.6:5.6]
set ylabel "Road phase"
set ytics ("HIGHWAY" 0, "LANE_CHANGE" 1, "EXIT" 2, "STOP_SIGN" 3, "YIELD" 4, "MOVE" 5)
set grid xtics ytics
unset key
phase(s) = s eq "HIGHWAY" ? 0 : s eq "LANE_CHANGE" ? 1 : s eq "EXIT" ? 2 : s eq "STOP_SIGN" ? 3 : s eq "YIELD" ? 4 : 5
plot csv every ::1 using (($0-1)*dt):(phase(stringcolumn(1))) with steps linewidth 4 linecolor rgb "#3366cc"

set yrange [-0.15:5.95]
set xlabel "Time (seconds)"
set ylabel "Car-presence signals"
set ytics ("Cross right" 0, "Cross left" 1, "Right front" 2, "Right side" 3, "Right rear" 4, "Front car" 5)
set grid xtics ytics
set key outside right center
plot \
    csv every ::1 using (($0-1)*dt):(5 + column(2)*0.7) with filledsteps title "Front car" linecolor rgb "#d62728", \
    csv every ::1 using (($0-1)*dt):(4 + (column(3) && stringcolumn(4) eq "RIGHT_REAR")*0.7) with filledsteps title "Right rear" linecolor rgb "#ff7f0e", \
    csv every ::1 using (($0-1)*dt):(3 + (column(3) && stringcolumn(4) eq "RIGHT_SIDE")*0.7) with filledsteps title "Right side" linecolor rgb "#9467bd", \
    csv every ::1 using (($0-1)*dt):(2 + (column(3) && stringcolumn(4) eq "RIGHT_FRONT")*0.7) with filledsteps title "Right front" linecolor rgb "#8c564b", \
    csv every ::1 using (($0-1)*dt):(1 + column(5)*0.7) with filledsteps title "Cross left" linecolor rgb "#2ca02c", \
    csv every ::1 using (($0-1)*dt):(column(6)*0.7) with filledsteps title "Cross right" linecolor rgb "#17becf"

unset multiplot
