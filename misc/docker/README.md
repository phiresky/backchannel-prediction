Run via

    docker run -p 3000:3000 -p 8765:8765 -ti phiresky/backchanneler-live-demo

then visit <http://localhost:3000> and wait a bit.

Build via

    git clone git@bitbucket.org:jrtk/janus.git
    wget https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
    mkdir adc
    for f in sw2249.wav sw2254.wav sw2258.wav sw2297.wav sw2411.wav sw2432.wav sw2485.wav sw2606.wav sw2735.wav sw2762.wav sw4193.wav sw2807.wav; do
        cp ../data/adc/$f ./adc/$f
    done
    docker build -t phiresky/backchanneler-live-demo .

