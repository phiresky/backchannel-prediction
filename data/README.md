

You will need the switchboard audio files from <https://catalog.ldc.upenn.edu/ldc97s62>

If you are running on the ISL cluster:

	ln -s /project/earsData/swbLinks adc

The ISL utterance database (should not be needed anymore, replaced by the ISIP transcriptions)

	ln -s /project/ears2/db/train/ db

The original transcriptions are from <https://www.isip.piconepress.com/projects/switchboard/>

	wget https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
	tar xf switchboard_word_alignments.tar.gz
	
You can compare the integrity of the data files with the ones I used by running

    sha256sum -c sha256sums.txt