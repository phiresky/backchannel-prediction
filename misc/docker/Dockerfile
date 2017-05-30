FROM base/devel

RUN pacman -Sy --noconfirm yarn python-pip libsndfile git openssh cmake tcl cython

COPY janus-sshkey /root/.ssh/id-rsa
COPY janusrc /root/.janusrc
COPY janus /janus
COPY .. /code

WORKDIR /janus
RUN mkdir build
RUN cd build
RUN cmake ..
RUN make -j$(nproc)
RUN python setup.py install

WORKDIR /code/web_vis/ts
RUN yarn
RUN yarn run dev

WORKDIR /code
RUN scripts/getdata.sh

RUN pip install websockets soundfile tqdm joblib
# todo: specific version
RUN pip install --upgrade https://github.com/Theano/Theano/archive/c697eeab84e5b8a74908da654b66ec9eca4f1291.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/b1e5bc468a2fbc5e5d026f6d1c6170b80e8be224.zip

RUN python3 -m web_vis.py.server configs/finunified/vary-features/lstm-best-features-raw_power,pitch,ffv.json

EXPOSE 3000
EXPOSE 8765
