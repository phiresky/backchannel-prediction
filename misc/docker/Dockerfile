FROM base/devel

RUN pacman -Sy --noconfirm yarn python-pip libsndfile git cmake tcl cython

RUN pip install websockets soundfile tqdm joblib
RUN pip install --upgrade https://github.com/Theano/Theano/archive/c697eeab84e5b8a74908da654b66ec9eca4f1291.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/b1e5bc468a2fbc5e5d026f6d1c6170b80e8be224.zip

COPY janusrc /root/.janusrc
COPY janus/.git /janus/.git

WORKDIR /janus
RUN git reset --hard 5d6fb82f5b1ca71b6ca4a96cc9cf6ebcbecbf6de
RUN mkdir build && cd build && cmake ..
WORKDIR /janus/build
RUN make -j$(nproc)
RUN python setup.py develop

RUN git clone --branch v054-docker-v3 --depth 1  https://github.com/phiresky/backchannel-prediction /code

WORKDIR /code/web_vis/ts
RUN yarn

WORKDIR /code/data
COPY switchboard_word_alignments.tar.gz /code/data
RUN tar xf switchboard_word_alignments.tar.gz && rm switchboard_word_alignments.tar.gz

COPY adc /code/data/adc

EXPOSE 3000
EXPOSE 8765

RUN echo 'en_US.UTF-8 UTF-8' > /etc/locale.gen \
     && locale-gen \
     && echo 'LANG=en_US.UTF-8' > /etc/locale.conf
ENV LANG en_US.UTF-8

WORKDIR /code
# fill caches
RUN python -c 'from web_vis.py.server import fill_caches; fill_caches("configs/finunified/vary-features/lstm-best-features-raw_power,pitch,ffv.json")'

CMD python3 -m web_vis.py.server configs/finunified/vary-features/lstm-best-features-raw_power,pitch,ffv.json & \
 (cd /code/web_vis/ts && yarn run dev)
