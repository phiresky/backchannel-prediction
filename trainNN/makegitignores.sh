# ignore all epoch files except the best one
for f in out/*; do
	if [[ -f "$f/config.json" ]]; then
		best=$(jq -r '[.train_output.stats[]]|min_by(.validation_error).weights' $f/config.json)
		if [[ $? -eq 0 ]]; then
			cat <<EOF >$f/.gitignore
*.pkl
!$best
EOF
		fi
	fi
done
