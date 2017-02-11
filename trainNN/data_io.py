import pickle
import logging


# Save current network parameters
def save_network_params(network, dumpfile, val_error=None):
    import lasagne
    logging.info("Saving network params to {}".format(dumpfile))

    try:
        param_values = lasagne.layers.get_all_param_values(network)
        d = {'param_values': param_values}
        if val_error is not None:
            d['val_error'] = val_error
        fd = open(dumpfile, 'wb')
        pickle.dump(d, fd, protocol=pickle.HIGHEST_PROTOCOL)
        fd.close()
    except:
        logging.info("Error saving network state to '{}'.".format(dumpfile))


# Restore network parameters
def load_network_params(network, dumpfile):
    import lasagne
    logging.info("Loading old params from %s" % dumpfile)

    fd = open(dumpfile, 'rb')
    d = pickle.load(fd)
    try:
        lasagne.layers.set_all_param_values(network, d['param values'])
    except:
        lasagne.layers.set_all_param_values(network, d['param_values'])
    fd.close()

    if 'val_error' in d:
        return d['val_error']
    else:
        return None
