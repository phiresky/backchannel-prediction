for f in trainNN/out/*/config.json; do
	version="$(basename "$(dirname "$f")")"
	[[ -d "evaluate/out/$version" ]] || echo "$f"
done
