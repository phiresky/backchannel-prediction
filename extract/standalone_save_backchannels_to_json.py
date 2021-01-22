import logging
from extract import readDB
from pathlib import Path
import json

def go():
    logging.root.handlers.clear()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[
                            # logging.FileHandler(LOGFILE),
                            logging.StreamHandler()
                        ])
    logging.info("hello")
    # shouldn't really matter except for some paths
    config_path = "configs/finunified/lstm-best-features-offline.json"
    config = readDB.util.load_config(config_path)
    convo_map = readDB.read_conversations(config)
    allconvos = [convo for convos in convo_map.values() for convo in convos]

    reader = readDB.loadDBReader(config_path)
    # specify this as used in the evaluation so we can distinguish between them
    reader.config["extract_config"]["min_talk_len"] = 5
    out = []
    for convo in allconvos:
        for channel in ["A", "B"]:
            convid = f"{convo}-{channel}"
            utts = list(reader.get_utterances(convid))
            for index, (uttId, uttInfo) in enumerate(utts):
                state = "non-bc"
                if reader.is_backchannel(uttInfo, index, utts):
                    if readDB.bc_is_while_monologuing(reader, uttInfo):
                        state = "monologuing-bc"
                    else:
                        state = "dialog-bc"
                out.append((uttId, state))
    out.sort()
    out = dict(out)
    with Path('data/utterance_is_backchannel.json').open("w") as f:
        json.dump(out, f, indent="\t")
    
    logging.info("done")


go()