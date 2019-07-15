from tensorflow import keras


def build_and_compile_model(hp, x):
    inputs = keras.layers.Input(shape=x.shape[1:])
    net = keras.layers.Dense(units=hp.num_neurons, activation=hp.activation)(inputs)
    for _ in range(1, hp.num_layers):
        net = keras.layers.Dense(units=hp.num_neurons, activation=hp.activation)(net)
        if hp.dropout_rate > 0:
            net = keras.layers.Dropout(rate=hp.dropout_rate)(net)
    net = keras.layers.Dense(hp.num_classes, activation='softmax')(net)
    model = keras.models.Model(inputs=inputs, outputs=net)
    model.compile(
        optimizer=keras.optimizers.Adadelta(lr=hp.learning_rate),
        loss=keras.losses.categorical_crossentropy,
        metrics=hp.metrics,
    )
    return model

