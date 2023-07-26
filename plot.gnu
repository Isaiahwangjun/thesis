set term pdf dashed
set output '/home/wang/Desktop/thesis/final_result/original/fig/test_21_f1.pdf'

set yrange [0:100]
set style data boxes
set style fill solid
set boxwidth 0.3
set xtics format ""
set grid ytics
set tics font "Times New Roman, 10"
set xtics out scale 0.5 nomirror 

set title "Test-21 F1-score (%)" font "Times New Roman, 14"
plot "/home/wang/Desktop/thesis/final_result/original/test_21_f1" using 2:xtic(1) notitle lt rgb "#272727",   \
     "" using 3 notitle lt rgb "#00FF00",   \
     "" using 4 notitle ,    \
     "" using 5 notitle ,    \
     "" using 6 notitle ,    \
     "" using 7 notitle ,    \
     "" using 8 notitle ,    \
     "" using 9 notitle ,    \
     "" using 10 notitle ,    \
     "" using 11 notitle 