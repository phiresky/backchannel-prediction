set -e

for config in "$@"; do
	confname="$(basename "$config")"
	echo batching $config
	sbatch --mem 4000M -p lowGPU,highGPU -J "$confname" --gres=gpu:1 ./DO.srun.sh trainNN.train "$config"
done
