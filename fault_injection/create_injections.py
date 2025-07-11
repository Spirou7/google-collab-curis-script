import tensorflow as tf
from local_tpu_resolver import LocalTPUClusterResolver

from models.resnet import resnet_18
from models.backward_resnet import backward_resnet_18
from models.resnet_nobn import resnet_18_nobn
from models.backward_resnet_nobn import backward_resnet_18_nobn
from models import efficientnet
from models import backward_efficientnet
from models import densenet
from models import backward_densenet
from models import nf_resnet
from models import backward_nf_resnet

import config
from prepare_data import generate_datasets
import math
import os
import argparse
import numpy as np
from models.inject_utils import *
from injection import Injection
import random
import struct


tf.config.set_soft_device_placement(True)
tf.random.set_seed(123)

golden_grad_idx = {
    'resnet18': -2,
    'resnet18_nobn': -2,
    'resnet18_sgd': -2,
    'effnet': -4,
    'densenet': -2,
    'nfnet': -2
    }


def parse_args():
    desc = "Tensorflow implementation of Resnet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file', type=str, help="Choose a csv file to replay")
    return parser.parse_args()


def get_model(m_name, seed):
    if m_name == 'resnet18' or m_name == 'resnet18_sgd':
        model = resnet_18(seed, m_name)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18(m_name)

    elif m_name == 'resnet18_nobn':
        model = resnet_18_nobn(seed)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18_nobn()

    elif m_name == 'effnet':
        model = efficientnet.efficient_net_b0(seed)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_efficientnet.backward_efficient_net_b0()

    elif m_name == 'densenet':
        model = densenet.densenet_121(seed)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_densenet.backward_densenet_121()

    elif m_name == 'nfnet':
        model = nf_resnet.NF_ResNet(num_classes=10, seed=seed, alpha=1, stochdepth_rate=0)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_nf_resnet.BackwardNF_ResNet(num_classes=10, alpha=1, stochdepth_rate=0)

    return model, back_model

def _get_injectable_layers_recursive(model):
    injectable_layers = []
    for layer in model.layers:
        # Some layers might not have 'l_name', and we are only interested in layers that are injectable
        if hasattr(layer, 'l_name') and layer.l_name is not None:
            injectable_layers.append(layer)
        if hasattr(layer, 'layers') and layer.layers is not None:
            injectable_layers.extend(_get_injectable_layers_recursive(layer))
    return injectable_layers

def bin2fp32(bin_str):
    assert len(bin_str) == 32
    return struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]

def fp322bin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))

def get_random_value():
    np.random.seed(None)
    one_bin = ''
    result = 0
    # Ensure that the generated random value is a finite number
    while one_bin == '' or not np.isfinite(result):
        one_bin = ''
        for _ in range(32):
            one_bin += str(np.random.randint(0,2))
        result = struct.unpack('!f',struct.pack('!I', int(one_bin, 2)))[0]
    return result



def flip_one_bit(target):
    np.random.seed(None)
    bin_target = fp322bin(target)
    #
    flip = np.random.randint(32)
    bin_output = ""
    for i in range(32):
        if i == flip:
            bin_output += ('1' if bin_target[i] == '0' else '0')
        else:
            bin_output += bin_target[i]
    return bin2fp32(bin_output)



def get_random_correct(target):
    shape = target.shape
    # get a random position from the flattened tensor
    rd_pos = np.unravel_index(np.random.randint(np.prod(shape)), shape)
    return target[rd_pos]



# generate random injection parameters
def get_rand_inj_constraints():
    """
    Constructs an Injection object with random values.
    """
    # 1. Initialize TPU Strategy
    resolver = LocalTPUClusterResolver()
    try:
        tf.tpu.experimental.initialize_tpu_system(resolver)
    except ValueError:
        pass # TPU already initialized
    strategy = tf.distribute.TPUStrategy(resolver)
    
    # 2. Randomly select a model
    model_names = ['resnet18', 'resnet18_sgd', 'resnet18_nobn', 'effnet', 'densenet', 'nfnet']
    model_name = random.choice(model_names)
    seed = random.randint(0, 2**32 - 1)
    
    with strategy.scope():
        model, back_model = get_model(model_name, seed)

    # 3. Generate other random parameters
    target_epoch = random.randint(1, 50)
    target_step = random.randint(1, 1000)

    inj = Injection()

    # 1. Set model name
    inj.model = model_name

    # 2. Set stage and injection type (fmodel)
    inj.stage = random.choice(['fwrd_inject', 'bkwd_inject'])
    
    inj_type_enum = random.choice(list(InjType))
    inj.fmodel = inj_type_enum.name

    # For now, only support weight target injections, as other types might require a forward pass
    while not is_weight_target(inj_type_enum):
        inj_type_enum = random.choice(list(InjType))
        inj.fmodel = inj_type_enum.name

    # 3. Set target worker
    inj.target_worker = random.randint(0, strategy.num_replicas_in_sync - 1)

    # 4. Set target layer
    injectable_layers = _get_injectable_layers_recursive(model)
    if not injectable_layers:
        raise ValueError("No injectable layers found in the model.")
    
    target_layer_obj = random.choice(injectable_layers)
    
    # Ensure the chosen layer has weights, which is necessary for weight-based injections
    while not target_layer_obj.get_weights():
        target_layer_obj = random.choice(injectable_layers)
        
    inj.target_layer = target_layer_obj.l_name

    inj.target_epoch = target_epoch
    inj.target_step = target_step

    # 5. Set injection positions and values
    weights = target_layer_obj.get_weights()
    
    target_tensor = weights[0] # Assuming kernel is the first weight
    shape = target_tensor.shape

    n_inj, n_repeat = num_inj(inj_type_enum)
    
    total_elements = np.prod(shape)
    if total_elements == 0:
        raise ValueError(f"Cannot inject into a tensor of size 0 for layer {inj.target_layer}")

    num_positions_to_generate = n_inj * n_repeat
    if total_elements < num_positions_to_generate:
        num_positions_to_generate = int(total_elements)

    flat_indices = np.random.choice(total_elements, size=num_positions_to_generate, replace=False)
    positions = [np.unravel_index(i, shape) for i in flat_indices]
    
    inj.inj_pos = [list(p) for p in positions]

    values = []
    for pos_tuple in positions:
        if is_random(inj_type_enum):
            val_delta = get_random_value()
        elif is_bflip(inj_type_enum):
            ori_val = target_tensor[pos_tuple]
            val_delta = flip_one_bit(ori_val)
        elif is_correct(inj_type_enum):
            val_delta = get_random_correct(target_tensor)
        else: # is_zero()
            val_delta = 0.0
        values.append(val_delta)
    
    inj.inj_values = values

    # 6. Set learning rate and seed
    inj.learning_rate = random.choice([0.001, 0.0005, 0.0001]) # common learning rates
    inj.seed = random.randint(0, 2**32 - 1)

    return inj, model, back_model

def main():
    args = parse_args()
    if args is None:
        exit()

    # TPU settings
    tpu_name = os.getenv('TPU_NAME')
    resolver = LocalTPUClusterResolver()
    tf.tpu.experimental.initialize_tpu_system(resolver)

    strategy = tf.distribute.TPUStrategy(resolver)
    per_replica_batch_size = config.BATCH_SIZE // strategy.num_replicas_in_sync
    print("Finish TPU strategy setting!")

    inj, model, back_model = get_rand_inj_constraints()
    inj.seed = 123

    # get the dataset
    train_dataset, valid_dataset, train_count, valid_count = generate_datasets(inj.seed)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    with strategy.scope():
        model, back_model = get_model(inj.model, inj.seed)
        # define loss and optimizer
        if 'sgd' in inj.model:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            	initial_learning_rate=inj.learning_rate,
            	decay_steps = 2000,
            	end_learning_rate=0.001)
            model.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        elif 'effnet' in inj.model:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=inj.learning_rate,
                decay_steps = 2000,
                end_learning_rate=0.0005)
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=inj.learning_rate,
                decay_steps = 5000,
                end_learning_rate=0.0001)
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(iterator):
        def step_fn(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs, _, _, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)

            tvars = model.trainable_variables
            gradients = tape.gradient(avg_loss, tvars)
            model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))

            train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
            train_accuracy.update_state(labels, predictions)
            return avg_loss

        return strategy.run(step_fn, args=(next(iterator),))


    @tf.function
    def fwrd_inj_train_step1(iter_inputs, inj_layer):
        def step1_fn(inputs):
            images, labels = inputs
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
            predictions = outputs['logits']
            return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]
        return strategy.run(step1_fn, args=(iter_inputs,))

    @tf.function
    def fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag):
        def step2_fn(inputs, inject):
            with tf.GradientTape() as tape:
                images, labels = inputs
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=inject, inj_args=inj_args)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)

            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model(man_grad_start, l_inputs, l_kernels)

            gradients = manual_gradients + golden_gradients[golden_grad_idx[inj.model]:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))

            train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
            train_accuracy.update_state(labels, predictions)
            return avg_loss

        return strategy.run(step2_fn, args=(iter_inputs, inj_flag))

    @tf.function
    def bkwd_inj_train_step1(iter_inputs, inj_layer):
        def step1_fn(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, _ = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start = tape.gradient(avg_loss, grad_start)
            _, bkwd_inputs, bkwd_kernels, bkwd_outputs = back_model(man_grad_start, l_inputs, l_kernels)
            return bkwd_inputs[inj_layer], bkwd_kernels[inj_layer], bkwd_outputs[inj_layer]

        return strategy.run(step1_fn, args=(iter_inputs,))

    @tf.function
    def bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag):
        def step2_fn(inputs, inject):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model(man_grad_start, l_inputs, l_kernels, inject=inject, inj_args=inj_args)

            gradients = manual_gradients + golden_gradients[golden_grad_idx[inj.model]:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))

            train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
            train_accuracy.update_state(labels, predictions)

            return avg_loss

        return strategy.run(step2_fn, args=(iter_inputs, inj_flag))


    @tf.function
    def valid_step(iterator):
        def step_fn(inputs):
            images, labels = inputs
            outputs , _, _, _ = model(images, training=False)
            predictions = outputs['logits']
            v_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            v_loss = tf.nn.compute_average_loss(v_loss, global_batch_size=config.BATCH_SIZE)
            valid_loss.update_state(v_loss)
            valid_accuracy.update_state(labels, predictions)
        return strategy.run(step_fn, args=(next(iterator),))

    steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
    valid_steps_per_epoch = math.ceil(valid_count / config.VALID_BATCH_SIZE)
 
    target_epoch = inj.target_epoch
    target_step = inj.target_step

    train_recorder = open("replay_{}.txt".format(args.file[args.file.rfind('/')+1:args.file.rfind('.')]), 'w')
    record(train_recorder, "Inject to epoch: {}\n".format(target_epoch))
    record(train_recorder, "Inject to step: {}\n".format(target_step))

    start_epoch = target_epoch
    total_epochs = config.EPOCHS
    early_terminate = False
    epoch = start_epoch
    while epoch < total_epochs:
        if early_terminate:
            break
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0

        train_iterator = iter(train_dataset)
        for step in range(steps_per_epoch):
            train_loss.reset_states()
            train_accuracy.reset_states()
            if early_terminate:
                break
            if epoch != target_epoch or step != target_step:
                losses = train_step(train_iterator)
            else:
                iter_inputs = next(train_iterator)
                inj_layer = inj.target_layer

                if 'fwrd' in inj.stage:
                    l_inputs, l_kernels, l_outputs = fwrd_inj_train_step1(iter_inputs, inj_layer)
                else:
                    l_inputs, l_kernels, l_outputs = bkwd_inj_train_step1(iter_inputs, inj_layer)

                inj_args, inj_flag = get_replay_args(InjType[inj.fmodel], inj, strategy, inj_layer, l_inputs, l_kernels, l_outputs, train_recorder)

                if 'fwrd' in inj.stage:
                    losses = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                else:
                    losses = bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag)

            record(train_recorder, "Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}\n".format(epoch,
                             total_epochs,
                             step,
                             steps_per_epoch,
                             train_loss.result(),
                             train_accuracy.result()))

            if not np.isfinite(train_loss.result()):
                record(train_recorder, "Encounter NaN! Terminate training!\n")
                early_terminate = True

        if not early_terminate:
            valid_iterator = iter(valid_dataset)
            for _ in range(valid_steps_per_epoch):
                valid_step(valid_iterator)

            record(train_recorder, "End of epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}\n".format(epoch,
                             config.EPOCHS,
                             train_loss.result(),
                             train_accuracy.result(),
                             valid_loss.result(),
                             valid_accuracy.result()))

            # NaN value in validation
            if not np.isfinite(valid_loss.result()):
                record(train_recorder, "Encounter NaN! Terminate training!\n")

                early_terminate = True

        epoch += 1


if __name__ == '__main__':
    main()
