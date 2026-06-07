conda activate scada


# OfficeHome Multi-Class SCADA-UL for C_F = {1,2,3}

# Our Method

python main.py -s "Art" -t "Clipart" -m "minimax" -fc 1,2,3
python main.py -s "Art" -t "Product" -m "minimax" -fc 1,2,3
python main.py -s "Art" -t "Real_World" -m "minimax" -fc 1,2,3
python main.py -s "Clipart" -t "Art" -m "minimax" -fc 1,2,3
python main.py -s "Clipart" -t "Product" -m "minimax" -fc 1,2,3
python main.py -s "Clipart" -t "Real_World" -m "minimax" -fc 1,2,3
python main.py -s "Product" -t "Art" -m "minimax" -fc 1,2,3
python main.py -s "Product" -t "Clipart" -m "minimax" -fc 1,2,3
python main.py -s "Product" -t "Real_World" -m "minimax" -fc 1,2,3
python main.py -s "Real_World" -t "Art" -m "minimax" -fc 1,2,3
python main.py -s "Real_World" -t "Clipart" -m "minimax" -fc 1,2,3
python main.py -s "Real_World" -t "Product" -m "minimax" -fc 1,2,3``