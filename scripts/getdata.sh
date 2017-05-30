cd "$(dirname "$0")"
cd "$(git rev-parse --show-toplevel)"
cd data

wget https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
tar xf switchboard_word_alignments.tar.gz

