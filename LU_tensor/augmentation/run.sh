for i in $(seq 0 29);
do
    python ./main.py CNN 0.001 100 128 0 15;
    python ./main.py LSTM 0.001 100 128 0 15;
    python ./main.py AT 0.001 100 128 0 15;
    python ./main.py EN 0.001 100 128 0 15;
    python ./main.py CNN 0.001 100 128 1 15;
    python ./main.py LSTM 0.001 100 128 1 15;
    python ./main.py AT 0.001 100 128 1 15;
    python ./main.py EN 0.001 100 128 1 15;
    python ./main.py CNN 0.001 100 128 2 15;
    python ./main.py LSTM 0.001 100 128 2 15;
    python ./main.py AT 0.001 100 128 2 15;
    python ./main.py EN 0.001 100 128 2 15
done