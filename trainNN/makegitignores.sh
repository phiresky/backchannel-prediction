for f in out/*; do
	best=$(jq -r '[.train_output.stats[]]|min_by(.validation_error).weights' $f/config.json)
	if [[ $? -eq 0 ]]; then
		cat <<EOF >$f/.gitignore
*.pkl
!$best
EOF
	fi
done
