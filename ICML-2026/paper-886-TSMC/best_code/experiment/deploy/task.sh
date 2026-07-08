#!/usr/bin/env bash

# Parse Console arguments
pass_through=""
while [ "$#" -gt 0 ]; do
  case $1 in
    -C|--config)
      # Capture variable length arguments to -C/ --config
      shift
      option_config=()
      while [ "$#" -gt 0 ] && [[ $1 != -* ]]; do
        option_config+=("$1")
        shift
      done
      ;;

    -P|--problem) option_problem="$2"; shift 2;;
    -O|--out) option_out="$2"; shift 2;;
    -M|--module) option_module="$2"; shift 2;;

    *) pass_through="$pass_through $1"; shift;;
  esac
done

# Throw away parsed CLI arguments
shift "$((OPTIND-1))"

# Check if Problem-Config is specified and add parsed configs to option_config
if [ -n "${option_problem+x}" ]; then

  # Format config database
  problem_configs=$(<"$option_problem")
  problem_configs=${problem_configs//$'\n'/ }
  problem_configs=${problem_configs//$'\r'/ }

  read -ra problem_elements <<< "$problem_configs"
  option_config+=("${problem_elements[*]}")
fi

# Join config-array to a single string for easier splitting.
config_str="${option_config[*]}"

# Run experiment with given specifications.
export PYTHONPATH="..:$PYTHONPATH"
python3 -m experiment.src.main  \
  -M $option_module \
  -C $config_str \
  -O "${option_out:-$HOME/out}" \
  $pass_through
