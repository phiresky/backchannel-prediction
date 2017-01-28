for conf in "$@"
do sbatch --partition=HPC,UBUNTU --job-name="eval-bc-$conf" --time=2-0 --cpus-per-task=48 --mem-per-cpu=6G ./DO.cpu.srun.sh evaluate.evaluate $conf
done
