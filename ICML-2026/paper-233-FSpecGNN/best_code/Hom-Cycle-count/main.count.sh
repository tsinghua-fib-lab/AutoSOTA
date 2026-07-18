export HOM="chordal4_1,g boat,g chordal6,g chordal4_4,v chordal4_1,v"
export HOM_e="chordal5_31,e chordal5_13,e chordal5_24,e"
export ISO="cycle3 cycle4 cycle5 cycle6 chordal4 chordal5"
export LEVEL="g v e"

device=1
seed=1
for task in $HOM; do
    python main.count.py --model Spec --task $(echo $task | tr "," " ") --device $device --seed $seed
done
for task in $HOM_e; do
    python main.count.py --model Spec --task $(echo $task | tr "," " ") --num_layer 3 --dim_embed 128 --device $device --seed $seed
done
for task in $ISO; do
    for level in $LEVEL; do
        python main.count.py --model Spec --task "$task" "$level" --device $device --seed $seed
    done
done
