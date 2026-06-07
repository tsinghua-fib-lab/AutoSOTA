#!/bin/bash
cd /repo

RESULTS_FILE="/repo/results_office31.txt"
echo "=== Office31 SCADA-UL Results ===" > $RESULTS_FILE
echo "" >> $RESULTS_FILE

TASKS=(
  "amazon dslr"
  "amazon webcam"
  "dslr amazon"
  "dslr webcam"
  "webcam amazon"
  "webcam dslr"
)

for TASK in "${TASKS[@]}"; do
  SRC=$(echo $TASK | cut -d' ' -f1)
  TGT=$(echo $TASK | cut -d' ' -f2)
  echo "=== Running: $SRC -> $TGT ===" | tee -a $RESULTS_FILE
  python main.py -d Office31 -s $SRC -t $TGT -m minimax -fc 1,2,3 -e 10 -se 10 2>&1 | tee -a $RESULTS_FILE
  echo "" >> $RESULTS_FILE
  echo "---" >> $RESULTS_FILE
  echo "" >> $RESULTS_FILE
done

echo "=== ALL TASKS COMPLETED ===" >> $RESULTS_FILE

# Print summary
echo ""
echo "=== SUMMARY ==="
grep -E "\[.*\].*minimax" $RESULTS_FILE
