import random
import logging
from collections import OrderedDict

import theano.tensor as T
import lasagne
import theano

from timeit import default_timer as timer
from .data_io import load_network_params, save_network_params


# Freeze layer
def freeze(layer):
    for param in layer.params:
        layer.params[param].discard('trainable')
    return layer


# Unfreeze layer
def unfreeze(layer):
    for param in layer.params:
        layer.params[param].add('trainable')
    return layer


sigterm_received = False


def sigterm(_signo, _stack_frame):
    global sigterm_received
    logging.info("received exit signal, waiting for epoch to finish...")
    sigterm_received = True


def train_network(network,
                  iterate_minibatches_train,
                  iterate_minibatches_validate,
                  categorical_output,
                  num_epochs,
                  output_prefix=None,
                  learning_rate_num=None,
                  update_method='nesterov',
                  scheduling_method=None,
                  scheduling_params=[0.005, 0.0001, 0.5],
                  max_norm=None,
                  momentum=0.9,
                  l2_regularization=None,
                  draw_theano_graph=None,
                  resume=None,
                  resume_error=None):
    import signal
    signal.signal(signal.SIGINT, sigterm)
    signal.signal(signal.SIGTERM, sigterm)
    outputFiles = []
    stats = OrderedDict()

    if scheduling_method is not None:
        logging.info("Using {} as schedulung method.".format(scheduling_method))

    if resume is not None:
        outputFiles.append(resume)
        load_network_params(network, resume)

    if categorical_output:
        target_var = T.ivector('targets')
    else:
        target_var = T.fvector('targets')
    learning_rate = T.fscalar('learning_rate')
    # States for newbob
    doScaleRate = False

    prediction = lasagne.layers.get_output(network)
    if categorical_output:
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    else:
        loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    if l2_regularization is not None:
        logging.debug("Applying L2 regularization with {:.6f}".format(l2_regularization))
        l2_penalty = lasagne.regularization.regularize_network_params(network,
                                                                      lasagne.regularization.l2) * l2_regularization
        loss = loss + l2_penalty

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if categorical_output:
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    else:
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    p_count = lasagne.layers.count_params(network)
    p_count_train = lasagne.layers.count_params(network, trainable=True)

    logging.info("Training network with {} trainable out of {} total params.".format(p_count_train, p_count))

    if update_method == 'nesterov':
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate)
        logging.debug("Using nesterov_momentum with learning_rate={:.6f}".format(learning_rate_num))
    elif update_method == 'momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
        logging.debug("Using sgd with learning_rate={:.6f} and momentum={:.6f}".format(learning_rate_num, momentum))
    elif update_method == 'sgd':
        if learning_rate_num is None:
            learning_rate_num = 1.0
        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        logging.debug("Using sgd with learning_rate={:.6f}".format(learning_rate_num))
    elif update_method == 'adadelta':
        if learning_rate_num is None:
            learning_rate_num = 1.0
        updates = lasagne.updates.adadelta(loss, params, learning_rate=learning_rate)
        logging.debug("Using adadelta with learning_rate={:.6f}".format(learning_rate_num))
    elif update_method == 'adam':
        if learning_rate_num is None:
            learning_rate_num = 0.001
        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
        logging.debug("Using adam with learning_rate={:.6f}".format(learning_rate_num))
    else:
        logging.debug("Error! {} is no valid update method.".format(update_method))

    if max_norm is not None:
        logging.debug("Applying max_norm={:.6f}".format(max_norm))
        for param in lasagne.layers.get_all_params(network, regularizable=True):
            updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm)

        logging.debug("Compiling theano functions...")

    input_var = lasagne.layers.get_all_layers(network)[0].input_var
    train_fn = theano.function([input_var, target_var, learning_rate], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if draw_theano_graph is not None:
        theano.printing.pydotprint(train_fn, outfile=draw_theano_graph, var_with_name_simple=True)

    logging.debug("Starting training...")

    if resume_error is None:
        oldValErr = 1
    else:
        oldValErr = resume_error

    oldLearningRate = learning_rate_num
    beginTime = timer()
    for epoch in range(num_epochs):
        if sigterm_received:
            break
        # Part 1: Train network
        train_err = 0
        train_batches = 0
        for inputs, outputs in iterate_minibatches_train():
            train_err += train_fn(inputs, outputs, learning_rate_num)
            train_batches += 1

        training_loss = train_err / train_batches

        if output_prefix is not None:
            dumpFile = "%s-%03d.pkl" % (output_prefix, epoch)
            outputFiles.append(dumpFile)
            save_network_params(network, dumpFile)

        # Part 2: Test network on validation set
        val_err = 0
        val_acc = 0
        val_batches = 0
        for inputs, outputs in iterate_minibatches_validate():
            err, acc = val_fn(inputs, outputs)
            val_err += err
            val_acc += acc
            val_batches += 1

        validation_loss = val_err / val_batches
        if categorical_output:
            validation_error = 1 - val_acc / val_batches
        else:
            validation_error = validation_loss

        if resume_error is None:
            resume_error = validation_error
            resume = dumpFile
        elif validation_error < resume_error:
            resume = dumpFile
            resume_error = validation_error

        endTime = timer()
        elapsed = endTime - beginTime
        beginTime = endTime
        logging.info(
            "epoch: {} took {:.3f}s\ntraining loss:\t{:.6f}\nvalidation loss:\t{:.6f}\nvalidation error:\t{:.6f}".format(
                epoch, elapsed, training_loss, validation_loss, validation_error))

        stats[epoch] = {
            'validation_error': validation_error,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'weights': dumpFile
        }
        delta_err = oldValErr - validation_error
        oldValErr = validation_error

        if scheduling_method is not None:
            if scheduling_method in ['newbob', 'soft_newbob']:
                nb_start, nb_end, nb_scale = scheduling_params

                if delta_err < nb_end:
                    if scheduling_method is 'newbob':
                        logging.info("Network below end threshold, exiting")
                        break
                elif delta_err < nb_start:
                    # The gains dropped below the start threshold, scale the learning rate
                    doScaleRate = True

                if doScaleRate:
                    learning_rate_num = learning_rate_num * nb_scale

                    # Reduce the learning rate only once during soft newbob
                    if scheduling_method is 'soft_newbob':
                        doScaleRate = False

            elif scheduling_method is 'exp':
                learning_rate_num = learning_rate_num * scheduling_params

            elif scheduling_method is 'fuzzy_newbob':
                nb_scale, nb_end = scheduling_params
                # Check if error increased
                fuzzy_delta = resume_error - validation_error
                if fuzzy_delta < 0:
                    # Decrease learning_rate
                    learning_rate_num = learning_rate_num * nb_scale
                    if learning_rate_num <= nb_end:
                        logging.info("learning rate below {}, ending".format(nb_end))
                        break
                    # Load old params into network
                    # load_network_params(network, outputFiles[-2])
                    load_network_params(network, resume)

            # Re-compile training function is learning rate has changed
            if oldLearningRate is not learning_rate_num:
                logging.info("{}: Updated {} with learning_rate={:.6f}".format(scheduling_method, update_method,
                                                                               learning_rate_num))

                logging.info("Re-compiling train function...")

                oldLearningRate = learning_rate_num
        yield stats


# Print network configuration
def print_network_config(input_var, net_dict):
    if isinstance(input_var, lasagne.layers.Layer):
        for key in list(net_dict.keys()):
            logging.info("{}\t{}".format(key, net_dict[key].num_units))
            var = net_dict[key].W.get_value()
            logging.info(
                "shape={}\tmin={:.6f}\tmax={:.6f}\tmean={:.6f}\tstd={:.6f}".format(var.shape, var.min(), var.max(),
                                                                                   var.mean(), var.std()))
            var = net_dict[key].b.get_value()
            logging.info(
                "shape={}\tmin={:.6f}\tmax={:.6f}\tmean={:.6f}\tstd={:.6f}".format(var.shape, var.min(), var.max(),
                                                                                   var.mean(), var.std()))

    elif isinstance(input_var, list):
        for var in input_var:
            logging.info(
                "shape={}\tmin={:.6f}\tmax={:.6f}\tmean={:.6f}\tstd={:.6f}".format(var.shape, var.min(), var.max(),
                                                                                   var.mean(), var.std()))

    else:
        logging.info("Error! Unsupported input.")
