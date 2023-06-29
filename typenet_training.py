# %%
import pandas as pd
import numpy as np

import math
import traceback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, BatchNormalization, Masking, Input

# %%
# Set seed for reproducibility
tf.keras.utils.set_random_seed(58)
# tf.debugging.enable_check_numerics()

# %%
# The paper says "For the _Contrastive_ loss we generate genuine and impostor pairs using all 
# the 15 keystroke sequences available for each subject". I assume that it is the same for triplet loss
enrollment_sequences_per_subject = 15

number_of_keystrokes_per_seq = 50  # also known as M in the paper
features_per_keystroke = 5
input_shape = (number_of_keystrokes_per_seq, features_per_keystroke)
batch_size = 512

input_dir = 'preprocessed_feather'

# %%
participants_sections = pd.read_table(
    f'{input_dir}/METADATA_participant_sections.tsv', header=None, index_col=0, names=['participant_id', 'sections'])


def parse_sections(sections):
    return np.array([int(section) for section in sections[1:-1].split(', ')])


participants_sections = participants_sections.dropna()

participants_sections.index = participants_sections.index.astype(int)

parsed_sections = participants_sections['sections'].apply(parse_sections)
parsed_sections

# %%
# it can be nice to have a deterministic order everytime and maybe randomise with seed later
# if the input is not sorted, the order of the training data may be different every time
# the preprocessing is done
participants = np.array(parsed_sections.index.sort_values())

# %%
train_participants = participants[:67000]
val_participants = participants[67000:68000]

test_participants = participants[68000:]

print(f"Train size: {train_participants.shape}")
print(f"Validation size: {val_participants.shape}")
print(f"Test size: {test_participants.shape}")

# %%
def zero_pad_rows(feature_dataframe, fixed_size_length):
    n_to_append = fixed_size_length - feature_dataframe.shape[0]
    if n_to_append > 0:
        feature_dataframe = pd.concat([feature_dataframe, pd.DataFrame(np.zeros(
            (n_to_append, feature_dataframe.shape[1])), columns=feature_dataframe.columns)])
    else:
        feature_dataframe = feature_dataframe.head(fixed_size_length)

    return feature_dataframe

# %%
class TripletIdSequence(tf.keras.utils.Sequence):
    """Generator of training triplet ids tuples (anchor, positive, negative)
        to be used by KeystrokeTripletSequence
    """

    def __init__(self, participant_ids, enrollment_sequences_per_subject):
        self.enrollment_sequences_per_subject = enrollment_sequences_per_subject
        self.participant_ids = participant_ids
        self.mapping = np.repeat(
            np.arange(len(self.participant_ids)), self.enrollment_sequences_per_subject)

    def __len__(self):
        return math.ceil(len(self.participant_ids)*self.enrollment_sequences_per_subject)

    def __getitem__(self, index):
        anchor_id = self.participant_ids[self.mapping[index]]
        anchor_sections = parsed_sections.loc[anchor_id]
        anchor_section = anchor_sections[index %
                                         self.enrollment_sequences_per_subject]

        positive_id = anchor_id

        # we remove the anchor from the positive or the ap_distance would be zero for that sample, which 
        # is bad for the triplet loss and could get the loss stuck at the margin more easily
        positive_sections_without_anchor_section = parsed_sections.loc[positive_id][parsed_sections.loc[positive_id] != anchor_section]
        positive_section = np.random.choice(positive_sections_without_anchor_section)

        negative_id = np.random.choice(self.participant_ids)
        while negative_id == anchor_id:
            negative_id = np.random.choice(self.participant_ids)
        negative_section = np.random.choice(
            parsed_sections.loc[negative_id])

        return (anchor_id, anchor_section), (positive_id, positive_section), (negative_id, negative_section)


# %%
from itertools import islice
for x in islice(TripletIdSequence(train_participants, enrollment_sequences_per_subject), 20):
    print(x)

# %%
class KeystrokeTripletSequence(tf.keras.utils.Sequence):

    def __init__(self, batch_size, participant_ids):
        self.batch_size = batch_size
        self.indices = TripletIdSequence(
            participant_ids, enrollment_sequences_per_subject)
        self.participant_ids = participant_ids

    def __len__(self):
        return math.ceil(len(self.participant_ids)*enrollment_sequences_per_subject / self.batch_size)

    def __getitem__(self, generator_idx):
        anchor_batch = np.empty((self.batch_size, *input_shape))
        positive_batch = np.empty((self.batch_size, *input_shape))
        negative_batch = np.empty((self.batch_size, *input_shape))

        for i in range(0, batch_size):
            (anchor_id, anchor_section_id), (positive_id, positive_section_id), (
                negative_id, negative_section_id) = self.indices[generator_idx]

            try:
                cols = ['KEYCODE', 'HL', 'IL', 'PL', 'RL']
                anchor_features = pd.read_feather(
                    f'{input_dir}/{anchor_id}/{anchor_id}_section_{anchor_section_id}_keystrokes.feather', columns=cols)
                positive_features = pd.read_feather(
                    f'{input_dir}/{positive_id}/{positive_id}_section_{positive_section_id}_keystrokes.feather', columns=cols)
                negative_features = pd.read_feather(
                    f'{input_dir}/{negative_id}/{negative_id}_section_{negative_section_id}_keystrokes.feather', columns=cols)

                padded_anchor_features = zero_pad_rows(
                    anchor_features, number_of_keystrokes_per_seq).values
                padded_positive_features = zero_pad_rows(
                    positive_features, number_of_keystrokes_per_seq).values
                padded_negative_features = zero_pad_rows(
                    negative_features, number_of_keystrokes_per_seq).values

                # Append to the batch
                anchor_batch[i] = np.array(padded_anchor_features)
                positive_batch[i] = np.array(padded_positive_features)
                negative_batch[i] = np.array(padded_negative_features)

            except Exception as e:
                print(e)
                traceback.print_exc()
                print(
                    f"Error with {anchor_id}, {positive_id}, {negative_id}, sections {anchor_section_id}, {positive_section_id}, {negative_section_id}")
                continue

        # warning: keras expects a y value too even if we dont have it
        # zeros = np.zeros((batch_size,) + input_shape)
        # return tuple like this [zeros,zeros,zeros], []
        return [anchor_batch, positive_batch, negative_batch], []


# %%
# Typenet embedding architecture
embedding = Sequential(name='embedding_rnn')
embedding.add(Input(shape=input_shape))
embedding.add(Masking(mask_value=0.0))
embedding.add(BatchNormalization())
embedding.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2))
embedding.add(Dropout(0.5))
embedding.add(BatchNormalization())
embedding.add(LSTM(128, recurrent_dropout=0.2))
embedding.add(Dropout(0.5))

embedding.summary()

# %%
# Distance layer code from:
# https://keras.io/examples/vision/siamese_network/


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=input_shape)
positive_input = layers.Input(name="positive", shape=input_shape)
negative_input = layers.Input(name="negative", shape=input_shape)

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

siamese_network.summary()

# %%
tf.keras.utils.plot_model(
    siamese_network, show_shapes=True, show_layer_names=True)

# %%
# Siamese code from:
# https://keras.io/examples/vision/siamese_network/


class TypeNet(keras.Model):

    def __init__(self, siamese_network, margin=1.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # ap: anchor positive
        # an: anchor negative
        ap_distance, an_distance = self.siamese_network(data)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

# FIXME: the margin parameter set here is ignored, only the default value for the parameter is considered
siamese_model = TypeNet(siamese_network, 1.5)

if __name__ == '__main__':
    # %%
    # resume training:
    #siamese_model.load_weights('checkpoint-10-0.538-0.992')

    # %%
    # 0.05 is stated in the paper but I found it to get stuck at the margin value while training
    learning_rate = 0.01
    
    optim = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    siamese_model.compile(optimizer=optim)

    # %%
    train_dataset = KeystrokeTripletSequence(batch_size, train_participants)
    val_dataset = KeystrokeTripletSequence(batch_size, val_participants)

    # check that I am not using the same subject in train, test and val
    print(set(train_participants) & set(test_participants))
    print(set(test_participants) & set(val_participants))
    print(set(train_participants) & set(val_participants))

    # %%
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "checkpoint-{epoch:02d}-{loss:.3f}-{val_loss:.3f}", save_best_only=False, monitor="val_loss", save_format='tf'
        ),
        #keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1, restore_best_weights=True),
        # profile_batch='20,24'),
        keras.callbacks.TensorBoard(log_dir=f"./logs-tensorboard"),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                 patience=2, min_lr=0.0001, verbose=1)
    ]

    # TODO: use_multiprocessing is toggled off because some data structure concurrent access bug
    # but should be easy to fix. It is going to fail after the second epoch IIRC.
    history = siamese_model.fit(
        train_dataset,
        epochs=200,
        callbacks=callbacks,
        validation_data=val_dataset,
        initial_epoch=0,
        max_queue_size=8,
        workers=2,
        use_multiprocessing=False,
        shuffle=True,
    )
