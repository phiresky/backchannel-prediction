set -e

for config in "$@"; do
	confname="$(dirname "$config")"
	echo batching $config
	sbatch --mem=10000M --partition=lowGPU,highGPU --job-name="$confname" --gres=gpu:1 --time=3-0 ./DO.srun.sh evaluate.evaluate "$config"
done
