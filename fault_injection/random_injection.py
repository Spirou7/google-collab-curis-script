import tensorflow as tf
import random
import numpy as np
import os
import csv
import glob
import datetime
import matplotlib.pyplot as plt
from models.inject_utils import choose_random_layer
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
from models.inject_utils import *
from injection import read_injection
import sys

tf.config.set_soft_device_placement(True)
# Keep seed consistent for reproducibility
tf.random.set_seed(123)

golden_grad_idx = {
    'resnet18': -2,
    'resnet18_nobn': -2,
    'resnet18_sgd': -2,
    'effnet': -4,
    'densenet': -2,
    'nfnet': -2
}


class RandomInjection:
    def __init__(self):
        self.model = ''
        self.stage = ''
        self.fmodel = ''
        self.target_worker = -1
        self.target_layer = ''
        self.target_epoch = -1
        self.target_step = -1
        self.inj_pos = []
        self.inj_values = []
        self.learning_rate = 0.001
        self.seed = 123
        
        # Available options based on CSV analysis
        self.available_models = ['resnet18', 'resnet18_sgd', 'resnet18_nobn', 'effnet', 'densenet', 'nfnet']
        self.available_stages = ['fwrd_inject', 'bkwd_inject']
        self.available_fmodels = ['INPUT', 'INPUT_16', 'WT', 'WT_16', "RBFLIP", "RD", "RD_CORRECT", "ZERO", "N16_RD", "N16_RD_CORRECT", "RD_GLB", "RD_CORRECT_GLB", "N64_INPUT", "N64_WT", "N64_INPUT_16", "N64_WT_16", "N64_INPUT_GLB", "N64_WT_GLB"]
        self.learning_rate_range = [0.0001, 0.001, 0.01, 0.1]
        self.min_val = None
        self.max_val = None
        
    def get_random_injection_params(self, model=None, stage=None, fmodel=None, 
                                  target_layer=None, target_epoch=None, 
                                  target_step=None, learning_rate=None,
                                  inj_pos=None, inj_values=None,
                                  min_val=None, max_val=None):
        """
        Generate random injection parameters. If parameters are provided, use them;
        otherwise generate random ones.
        """
        # Set parameters or generate random ones
        self.model = model if model else random.choice(self.available_models)
        self.stage = stage if stage else random.choice(self.available_stages)
        self.fmodel = fmodel if fmodel else random.choice(self.available_fmodels)
        self.target_worker = random.randint(1, 5)  # Based on CSV data
        self.target_layer = target_layer if target_layer else choose_random_layer(self.model, self.stage) 
        self.target_epoch = target_epoch
        self.target_step = target_step
        self.learning_rate = learning_rate if learning_rate else random.choice(self.learning_rate_range)
        
        # Injection position will be randomly selected by get_inj_args
        self.inj_pos = inj_pos if inj_pos is not None else "random"
        
        # Injection value will be randomly selected by get_inj_args
        self.inj_values = inj_values if inj_values is not None else "random"
            
        self.min_val = min_val
        self.max_val = max_val

        return {
            'model': self.model,
            'stage': self.stage,
            'fmodel': self.fmodel,
            'target_worker': self.target_worker,
            'target_layer': self.target_layer,
            'target_epoch': self.target_epoch,
            'target_step': self.target_step,
            'inj_pos': self.inj_pos,
            'inj_values': self.inj_values,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'min_val': self.min_val,
            'max_val': self.max_val
        }
    
    def save_injection_config(self, filename='random_injection_config.csv'):
        """Save the current injection configuration to a CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['model', self.model])
            writer.writerow(['stage', self.stage])
            writer.writerow(['fmodel', self.fmodel])
            writer.writerow(['target_worker', self.target_worker])
            writer.writerow(['target_layer', self.target_layer])
            writer.writerow(['target_epoch', self.target_epoch])
            writer.writerow(['target_step', self.target_step])
            writer.writerow(['inj_pos', self.inj_pos])
            writer.writerow(['inj_values', self.inj_values])
            writer.writerow(['learning_rate', self.learning_rate])
            writer.writerow(['seed', self.seed])
    
    def get_model(self, m_name, seed):
        """Get model and backward model based on reproduce_injections.py pattern"""
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

    def run_training_simulation(self):
        """Run the training simulation with the current injection parameters"""
        print(f"Running training simulation with parameters:")
        print(f"Model: {self.model}")
        print(f"Stage: {self.stage}")
        print(f"FModel: {self.fmodel}")
        print(f"Target Layer: {self.target_layer}")
        print(f"Target Epoch: {self.target_epoch}")
        print(f"Target Step: {self.target_step}")
        print(f"Injection Position: {self.inj_pos}")
        print(f"Injection Value: {self.inj_values}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Seed: {self.seed}")
        
        # Save config for reference
        self.save_injection_config()
        
        # Configure TensorFlow for CPU on MacOS
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(self.seed)
        
        # Get model and backward model
        model, back_model = self.get_model(self.model, self.seed)
        
        # Setup optimizer based on model type
        if 'sgd' in self.model:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=2000,
                end_learning_rate=0.001)
            model.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        elif 'effnet' in self.model:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=2000,
                end_learning_rate=0.0005)
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=5000,
                end_learning_rate=0.0001)
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Setup metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
        
        # Training functions (from reproduce_injections.py)
        @tf.function
        def train_step(iterator):
            images, labels = next(iterator)
            with tf.GradientTape() as tape:
                outputs, _, _, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)

            tvars = model.trainable_variables
            gradients = tape.gradient(avg_loss, tvars)
            model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))

            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss

        @tf.function
        def fwrd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
            predictions = outputs['logits']
            return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]

        def fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag):
            with tf.GradientTape() as tape:
                images, labels = iter_inputs
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=inj_flag, inj_args=inj_args)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)

            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs, l_kernels, inject=False, inj_args=None)

            gradients = manual_gradients + golden_gradients[golden_grad_idx[self.model]:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))

            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss

        def bkwd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, _ = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start = tape.gradient(avg_loss, grad_start)
            _, bkwd_inputs, bkwd_kernels, bkwd_outputs = back_model.call(man_grad_start, l_inputs, l_kernels, inject=False, inj_args=None)
            return bkwd_inputs[inj_layer], bkwd_kernels[inj_layer], bkwd_outputs[inj_layer]

        def bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag):
            images, labels = iter_inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs, l_kernels, inject=inj_flag, inj_args=inj_args)

            gradients = manual_gradients + golden_gradients[golden_grad_idx[self.model]:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))

            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss

        @tf.function
        def valid_step(iterator):
            images, labels = next(iterator)
            outputs, _, _, _ = model(images, training=False)
            predictions = outputs['logits']
            v_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            v_loss = tf.nn.compute_average_loss(v_loss, global_batch_size=config.BATCH_SIZE)
            valid_loss.update_state(v_loss)
            valid_accuracy.update_state(labels, predictions)

        # Training loop
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        valid_steps_per_epoch = math.ceil(valid_count / config.VALID_BATCH_SIZE)
        
        target_epoch = self.target_epoch
        target_step = self.target_step
        
        # Initialize plot
        plt.ion()
        fig, ax = plt.subplots()
        steps_history, accuracy_history = [], []
        line, = ax.plot(steps_history, accuracy_history, 'r-')
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Train Accuracy")
        ax.set_title("Real-time Training Accuracy")
        ax.grid(True)
        plt.show()

        # Create a timestamp and define file paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Setup plot and log paths
        plot_result_folder = "simulation_results/NaN"
        os.makedirs(plot_result_folder, exist_ok=True)
        plot_filename = f"{self.model}_{self.stage}_{self.fmodel}_{timestamp}_accuracy_plot.png"
        plot_path = os.path.join(plot_result_folder, plot_filename)
        
        temp_log_file = f"temp_log_{timestamp}.txt"
        train_recorder = open(temp_log_file, 'w')
        
        def record(recorder, text):
            recorder.write(text)
            recorder.flush()
            print(text.strip())
        
        record(train_recorder, f"Inject to epoch: {target_epoch}\n")
        record(train_recorder, f"Inject to step: {target_step}\n")
        
        start_epoch = 0
        total_epochs = config.EPOCHS
        early_terminate = False
        epoch = start_epoch
        
        while epoch < total_epochs:
            if early_terminate:
                break
            train_loss.reset_state()
            train_accuracy.reset_state()
            valid_loss.reset_state()
            valid_accuracy.reset_state()
            step = 0

            train_iterator = iter(train_dataset)
            for step in range(steps_per_epoch):
                print(f"epoch: {epoch}, step: {step}")
                if early_terminate:
                    break
                    
                if epoch != target_epoch or step != target_step:
                    losses = train_step(train_iterator)
                else:
                    print("performing injection!")
                    iter_inputs = next(train_iterator)
                    inj_layer = self.target_layer

                    if 'fwrd' in self.stage:
                        print("forward injection")
                        l_inputs, l_kernels, l_outputs = fwrd_inj_train_step1(iter_inputs, inj_layer)
                    else:
                        print("backward injection")
                        l_inputs, l_kernels, l_outputs = bkwd_inj_train_step1(iter_inputs, inj_layer)

                    # Create injection args with random position selection
                    if self.min_val is not None and self.max_val is not None:
                        print("performing random injection with min/max range!")
                        inj_args, inj_flag = get_inj_args_with_random_range(
                            InjType[self.fmodel], None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder,
                            self, self.min_val, self.max_val
                        )
                    elif self.inj_pos != "random" and self.inj_values != "random":
                        print("performing specific injection!")
                        inj_args, inj_flag = get_replay_args(
                            InjType[self.fmodel], self, None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder,
                            inj_pos=self.inj_pos, inj_values=self.inj_values
                        )
                    else:
                        print("performing random injection!")
                        inj_args, inj_flag = get_inj_args(InjType[self.fmodel], None, inj_layer, l_inputs, l_kernels, l_outputs, train_recorder, self)


                    if 'fwrd' in self.stage:
                        losses = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                    else:
                        losses = bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag)

                record(train_recorder, f"Epoch: {epoch}/{total_epochs}, step: {step}/{steps_per_epoch}, loss: {train_loss.result():.5f}, accuracy: {train_accuracy.result():.5f}\n")

                # Update plot
                global_step = epoch * steps_per_epoch + step
                steps_history.append(global_step)
                accuracy_history.append(train_accuracy.result().numpy())
                
                line.set_xdata(steps_history)
                line.set_ydata(accuracy_history)
                
                ax.relim()
                ax.autoscale_view()
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # Continuously save the plot
                fig.savefig(plot_path)

                if not np.isfinite(train_loss.result()):
                    record(train_recorder, "Encounter NaN! Terminate training!\n")
                    early_terminate = True

            if not early_terminate:
                valid_iterator = iter(valid_dataset)
                for _ in range(valid_steps_per_epoch):
                    valid_step(valid_iterator)

                record(train_recorder, f"End of epoch: {epoch}/{config.EPOCHS}, train loss: {train_loss.result():.5f}, train accuracy: {train_accuracy.result():.5f}, valid loss: {valid_loss.result():.5f}, valid accuracy: {valid_accuracy.result():.5f}\n")

            epoch += 1

        train_recorder.close()
        
        # Determine final log directory and filename
        if early_terminate:
            result_folder = "simulation_results/NaN"
        else:
            result_folder = "simulation_results/No_NaN"
            
        os.makedirs(result_folder, exist_ok=True)
        
        final_log_filename = f"{self.model}_{self.stage}_{self.fmodel}_{timestamp}.txt"
        final_log_path = os.path.join(result_folder, final_log_filename)
        
        # Move the log file
        os.rename(temp_log_file, final_log_path)

        # Plot is already saved, just announce and close
        print(f"Accuracy plot saved to {plot_path}")
        plt.close(fig)

        return {
            'final_train_accuracy': train_accuracy.result().numpy(),
            'final_train_loss': train_loss.result().numpy(),
            'final_valid_accuracy': valid_accuracy.result().numpy(),
            'final_valid_loss': valid_loss.result().numpy(),
            'early_terminate': early_terminate,
            'injection_params': self.get_random_injection_params()
        }


def random_fault_injection(model=None, stage=None, fmodel=None, 
                          target_layer=None, target_epoch=None, 
                          target_step=None, learning_rate=None,
                          inj_pos=None, inj_values=None,
                          min_val=None, max_val=None):
    """
    Main function to run random fault injection.
    If parameters are not provided, they will be randomly selected.
    """
    injector = RandomInjection()
    params = injector.get_random_injection_params(
        model=model, stage=stage, fmodel=fmodel,
        target_layer=target_layer, target_epoch=target_epoch,
        target_step=target_step, learning_rate=learning_rate,
        inj_pos=inj_pos, inj_values=inj_values,
        min_val=min_val, max_val=max_val
    )
    
    print("Generated Random Injection Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("\n" + "="*50 + "\n")
    
    # Run the simulation
    results = injector.run_training_simulation()
    
    return results


if __name__ == "__main__":
    # Example usage - completely random parameters
    print("Running with completely random parameters...")

    results = random_fault_injection(model='resnet18', target_epoch=0, target_step=15, stage='fwrd_inject', fmodel='INPUT', learning_rate=0.001, min_val=sys.float_info.max, max_val=sys.float_info.max)
    
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
