FROM base/devel

RUN pacman -Sy --noconfirm yarn python-pip libsndfile git openssh cmake tcl cython wget

RUN pip install websockets soundfile tqdm joblib
RUN pip install --upgrade https://github.com/Theano/Theano/archive/c697eeab84e5b8a74908da654b66ec9eca4f1291.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/b1e5bc468a2fbc5e5d026f6d1c6170b80e8be224.zip

COPY janusrc /root/.janusrc
COPY janus /janus

WORKDIR /janus
RUN mkdir build && cd build && cmake ..
WORKDIR /janus/build
RUN make -j$(nproc)
RUN python setup.py install

RUN git clone --branch v051-docker --depth 1 https://github.com/phiresky/backchannel-prediction /code

WORKDIR /code/web_vis/ts
RUN yarn

WORKDIR /code
RUN scripts/getdata.sh

COPY adc /code/data/adc

EXPOSE 3000
EXPOSE 8765

RUN echo 'en_US.UTF-8 UTF-8' > /etc/locale.gen \
     && locale-gen \
     && echo 'LANG=en_US.UTF-8' > /etc/locale.conf
ENV LANG en_US.UTF-8

CMD (cd /code && python3 -m web_vis.py.server configs/finunified/vary-features/lstm-best-features-raw_power,pitch,ffv.json)\
 & (cd /code/web_vis/ts && yarn run dev)
