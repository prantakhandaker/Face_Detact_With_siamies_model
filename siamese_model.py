from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Mean
from loss import contrastive_loss


class SiameseModel:
    def __init__(self, input_shape, embedding_dim, lr=1e-4):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.model = None

    def create_model(self):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        embedding_model = make_embedding_model(self.input_shape, self.embedding_dim)

        embedding_a = embedding_model(input_a)
        embedding_b = embedding_model(input_b)

        distance = Lambda(euclidean_distance)([embedding_a, embedding_b])

        self.model = Model(inputs=[input_a, input_b], outputs=distance)

    def compile_model(self):
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[Mean()])

    def train(self, train_generator, val_generator, epochs, steps_per_epoch, validation_steps, checkpoint_path):
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min')

        self.model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                       validation_data=val_generator, validation_steps=validation_steps, callbacks=[checkpoint])

    def evaluate(self, test_generator, steps):
        return self.model.evaluate(test_generator, steps=steps)
