import
import tensorflow as tf
from tqdm.keras import TqdmCallback


def _model():
    x = inputs = tf.keras.Input(shape=(...,))
    # DEFINE MODEL HERE

    # EXAMPLE: 
    # inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))
    # skip_list = []

    # filter_list = [16, 32, 64, 128, 256]

    # for filters in filter_list:
    #     x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
    #     x = tfa.layers.InstanceNormalization(axis=-1,
    #                                          center=True,
    #                                          scale=True,
    #                                          beta_initializer="random_uniform",
    #                                          gamma_initializer="random_uniform")(x)
    #     x = tf.keras.layers.ReLU()(x)
    #     x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
    #     x = tfa.layers.InstanceNormalization(axis=-1,
    #                                          center=True,
    #                                          scale=True,
    #                                          beta_initializer="random_uniform",
    #                                          gamma_initializer="random_uniform")(x)
    #     x = tf.keras.layers.ReLU()(x)
    #     skip_list.append(x)
    #     x = tf.keras.layers.MaxPool3D(2)(x)

    # x = tf.keras.layers.Conv3D(256, 3, **_ARGS)(x)
    # x = tf.keras.layers.Conv3D(256, 3, **_ARGS)(x)

    # for filters in reversed(filter_list):
    #     x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
    #     x = tf.keras.layers.concatenate([x, skip_list.pop()])
    #     x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
    #     x = tfa.layers.InstanceNormalization(axis=-1,
    #                                          center=True,
    #                                          scale=True,
    #                                          beta_initializer="random_uniform",
    #                                          gamma_initializer="random_uniform")(x)
    #     x = tf.keras.layers.ReLU()(x)

    #     x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
    #     x = tfa.layers.InstanceNormalization(axis=-1,
    #                                          center=True,
    #                                          scale=True,
    #                                          beta_initializer="random_uniform",
    #                                          gamma_initializer="random_uniform")(x)
    #     x = tf.keras.layers.ReLU()(x)

    # outputs = tf.keras.layers.Conv3D(2, 3, padding="same", activation="softmax")(x)

    # return tf.keras.Model(inputs, outputs)

    return tf.keras.Model(inputs=inputs, outputs=[..., ...])


def GENERATOR():
    # Define generator here 
    # i.e.
    # Read in difference map and regular map and input and output pdb files
    # Normalise each map 
    # For each residue calculate the RMSD to the deposited residue
    # Align each residue from bucaneer to a reference residue
    # Interpolate box of density (16,16,16) around each residue
    # Put regular density and difference density into one np array of shape (16,16,16,2)
    # Yield np.ndarray box and float rmsd

def train():
    # _train_gen = align.generate_dataset("train")
    # _test_gen = generate_dataset("test")

    # _train_gen = GENERATOR("train")
    # _test_gen = GENERATOR("test")


    input = tf.TensorSpec(shape=(2048,), dtype=tf.float32, name="input")
    output_sin = tf.TensorSpec(shape=(3), dtype=tf.float32, name="output_sin")
    output_cos = tf.TensorSpec(shape=(3), dtype=tf.float32, name="output_cos")

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, (output_sin, output_cos))
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, (output_sin, output_cos))
    )
    
    print(train_dataset)

    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = "test_1"

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)
    model = _model()
    model.summary()

    optimiser = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimiser, loss="mse",  metrics=['mse'])

    logger = tf.keras.callbacks.CSVLogger(f"train_{name}.csv", append=True)
    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{name}", histogram_freq=1, profile_batch=(10, 30)
    )
    
    callbacks_list = [
        checkpoint,
        # reduce_lr_on_plat,
        # TqdmCallback(verbose=2),
        # tensorboard_callback,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1,
        use_multiprocessing=True,
    )


if __name__ == "__main__":
    train()
