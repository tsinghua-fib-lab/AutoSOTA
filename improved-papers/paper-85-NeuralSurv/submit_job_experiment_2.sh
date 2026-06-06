#!/bin/sh

INDIR="/home/mm3218/git/neuralsurv"
OUTDIR="/home/mm3218/projects/2025/neuralsurv"

# Check if the environment deep_rl_liquidation exists
if conda env list | grep -q "^neuralsurv"; then
    echo "Conda environment neuralsurv found."
else
    echo "Conda environment neuralsurv not found. Please create it first."
    exit 1
fi

# Read JSON data from config.json
json_data=$(cat $INDIR/config_experiment_2.json)

# Remove line breaks from JSON data
json_data=$(echo "$json_data" | tr -d '\n')

# Parse JSON and process each entry
echo "$json_data" | grep -oP '\{[^}]*}' | while read -r entry; do

    # Extract job_name using grep (carefully crafted pattern)
    job_name=$(echo "$entry" | sed -n 's/.*"job_name":"\([^"]*\)".*/\1/p')

    # Job name with date
    JOBNAME=$(date +"%y%m%d")-${job_name}

    # Directory to folder
    CWD="$OUTDIR/${JOBNAME}"

    # Create folder
    mkdir -p "$CWD"

    # Create a JSON file with the entire entry (avoid parsing within loop)
    echo "$entry" > "$CWD/${JOBNAME}.json"

    # Create pbs for training score model
    cat > $CWD/${JOBNAME}.sh <<EOF
#!/bin/sh

source activate neuralsurv

INDIR=$INDIR
CWD=$CWD
JOBNAME=$JOBNAME

python \$INDIR/main.py --config=\$CWD/\$JOBNAME.json --workdir=\$CWD

EOF
done
