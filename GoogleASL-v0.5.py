import pyarrow.parquet as pq
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import numpy as np
import json
import os
import shutil
import tqdm
import glob
import logging
# print("Current working directory: {}".format(os.getcwd()))

# cwd = "/Users/arvinprince/pytorch-files/Google-ASL"
# print("Setting the working directory to : {}".format(cwd))
# os.chdir(cwd)
dataset_df = pd.read_csv("/Users/arvinprince/pytorch-files/Google-ASL/data/train.csv")

FRAME_LEN = 128
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f"x_right_hand_{i}" for i in range(21)] + [f"x_left_hand_{i}" for i in range(21)] + \
    [f"x_pose_{i}" for i in POSE]
Y = [f"y_right_hand_{i}" for i in range(21)] + [f"y_left_hand_{i}" for i in range(21)] + \
    [f"y_pose_{i}" for i in POSE]
Z = [f"z_right_hand_{i}" for i in range(21)] + [f"z_left_hand_{i}" for i in range(21)] + \
    [f"z_pose_{i}" for i in POSE]

FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]


def prepocessed_dir():
    if not os.path.isdir("/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed"):
        os.mkdir("/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed")
    else:
        shutil.rmtree("/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed")
        os.mkdir("/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed")
# Loop through each file_id
def processing_pq():
    bar = tqdm.tqdm(dataset_df["file_id"].unique(), desc="Parsing parquet files to TFRecord files")
    for file_id in bar:
        # Parquet file name
        '''
        pq_file = f"/Volumes/ExternalHDD/asl-fingerspelling/train_landmarks/{file_id}.parquet"
        '''
        # Filter train.csv and fetch entries only for the relevant file_id
        file_df = dataset_df.loc[dataset_df["file_id"] == file_id]
        # Fetch the parquet file
        parquet_df = pq.read_table(f"/Volumes/ExternalHDD/asl-fingerspelling/train_landmarks/{str(file_id)}.parquet",
                                   columns=['sequence_id'] + FEATURE_COLUMNS).to_pandas()
        # File name for the updated data
        tf_file = f"/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed/{file_id}.tfrecord"
        parquet_numpy = parquet_df.to_numpy()
        # Initialize the pointer to write the output of
        # each `for loop` below as a sequence into the file.
        with tf.io.TFRecordWriter(tf_file) as file_writer:
            # Loop through each sequence in file.
            for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
                # Fetch sequence data
                frames = parquet_numpy[parquet_df.index == seq_id]

                # Calculate the number of NaN values in each hand landmark
                r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
                l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
                no_nan = max(r_nonan, l_nonan)

                if 2 * len(phrase) < no_nan:
                    features = {FEATURE_COLUMNS[i]: tf.train.Feature(
                        float_list=tf.train.FloatList(value=frames[:, i])) for i in range(len(FEATURE_COLUMNS))}
                    features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                    record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                    file_writer.write(record_bytes)


# prepocessed_dir()
# processing_pq()

# tfrecords = dataset_df.file_id.map(lambda x: f"Google-ASL/data/preprocessed/{x}.tfrecord").unique()


tfrecords = glob.glob("/Users/arvinprince/pytorch-files/Google-ASL/data/preprocessed/*.tfrecord")
print("List of {} TFRecord Files".format(len(tfrecords)))

with open("/Users/arvinprince/pytorch-files/Google-ASL/data/character_to_prediction_index.json") as f:
    char_to_num = json.load(f)

pad_token = "P"
start_token = "<"
end_token = ">"
pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61

char_to_num[pad_token] = pad_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j: i for i, j in char_to_num.items()}


def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(tensor=x,
                   paddings=([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]),
                   constant_values=0,
                   )
    else:
        x = tf.image.resize(images=x,
                            size=(FRAME_LEN, tf.shape(x)[1]),
                            )
    return x


def preprocess(x):
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)

    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

    rnan = tf.math.count_nonzero(rnan_idx)
    lnan = tf.math.count_nonzero(lnan_idx)

    if rnan > lnan:
        hand = lhand
        pose = lpose
    else:
        hand = rhand
        pose = rpose

    hand_x = hand[:, 0 * (len(LHAND_IDX) // 3):1 * (len(LHAND_IDX) // 3)]
    hand_y = hand[:, 1 * (len(LHAND_IDX) // 3):2 * (len(LHAND_IDX) // 3)]
    hand_z = hand[:, 2 * (len(LHAND_IDX) // 3):3 * (len(LHAND_IDX) // 3)]

    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)

    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    pose_x = pose[:, 0 * (len(LPOSE_IDX) // 3):1 * (len(LPOSE_IDX) // 3)]
    pose_y = pose[:, 1 * (len(LPOSE_IDX) // 3):2 * (len(LPOSE_IDX) // 3)]
    pose_z = pose[:, 2 * (len(LPOSE_IDX) // 3):3 * (len(LPOSE_IDX) // 3)]

    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)

    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)

    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))

    return x


def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, features=schema)
    phrase = features["phrase"]
    landmarks = ([tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS])
    landmarks = tf.transpose(landmarks)
    return landmarks, phrase


table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=list(char_to_num.keys()),
        values=list(char_to_num.values())
    ),
    default_value=tf.constant(-1),
    name="class weights"
)


def convert_fn(landmarks, phrase):
    phrase = start_token + phrase + end_token
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    phrase = tf.pad(phrase, [[0, 64 - tf.shape(phrase)[0]]], mode="CONSTANT",
                    constant_values=pad_token_idx)
    return preprocess(landmarks), phrase


batch_size = 64
train_len = int(.8 * len(tfrecords))

train_ds = tf.data.TFRecordDataset(tfrecords[:train_len]).map(decode_fn).map(convert_fn) \
    .batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
val_ds = tf.data.TFRecordDataset(tfrecords[train_len:]).map(decode_fn).map(convert_fn) \
    .batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

M, n = next(iter(train_ds))


def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth

    angle_rates = 1 / (1e4**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1,
    )
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(length=max_len, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, *args, **kwargs):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
token_emb = PositionalEmbedding(vocab_size=100, max_len=64, d_model=512)
print(token_emb(n).shape)


class LandmarkEmbedding(keras.layers.Layer):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.conv1 = layers.Conv1D(
            d_model, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv1D(
            d_model, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv1D(
            d_model, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        pos_emb = positional_encoding(x.shape[1], self.d_model)
        return x +pos_emb
landmark_emb = LandmarkEmbedding(d_model=512)
print(landmark_emb(M).shape)


class BaseAttention(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.norm = layers.LayerNormalization()
        self.add = layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context, *args, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.norm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query = x,
            key = x,
            value =x
        )
        x = self.add([x, attn_output])
        x = self.norm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask = True,
        )
        x = self.add([x, attn_output])
        x = self.norm(x)
        return x


class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, x, *args, **kwargs):
        x = self.add([x, self.seq(x)])
        x = self.norm(x)
        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(self, *,
                 d_model:int,
                 num_heads: int,
                 dff:int,
                 dropout_rate=0.1,):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                  key_dim=d_model,
                                                  dropout= dropout_rate)
        self.ffn = FeedForward(d_model = d_model,
                               dff= dff,
                               dropout_rate=dropout_rate)

    def call(self, x, *args, **kwargs):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


sample_enc_layer = EncoderLayer(num_heads=2, d_model=512, dff=2048, dropout_rate=0.1)
print(sample_enc_layer(landmark_emb(M)).shape)


class Encoder(keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.landmark_emb = LandmarkEmbedding(d_model=d_model)
        self.enc_layer = [EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate
        ) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, *args, **kwargs):
        x = self.landmark_emb(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layer[i](x)
        return x
sample_encoder = Encoder(num_layers=2,
                         d_model=512,
                         dff=2048,
                         num_heads=8,
                         dropout_rate=.1)
print(sample_encoder(M, training=False).shape)

class DecoderLayer(keras.layers.Layer):
    def __init__(self, *,
                 num_heads, d_model, dff, dropout_rate=.1):
        super().__init__()
        self.causal_attention = CausalSelfAttention(num_heads=num_heads,
                                                  key_dim= d_model,
                                                  dropout= dropout_rate)
        self.cross_attention = CrossAttention(num_heads= num_heads,
                                              key_dim= d_model,
                                              dropout= dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_attention(x)
        x = self.cross_attention(x, context)

        self.last_attn_score = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x
sample_decoder_layer = DecoderLayer(num_heads=8, d_model=512, dff=2048)
sample_decoder_layer_output = sample_decoder_layer(landmark_emb(M), token_emb(n))
print(sample_decoder_layer_output.shape)


class Decoder(keras.layers.Layer):
    def __init__(self, *, num_layers, num_heads, max_len, d_model, dff,
                 vocab_size, dropout_rate=.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size, max_len, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.dec_layer = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.last_attention_score = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layer[i](x, context)
        self.last_attention_score = self.dec_layer[-1].last_attn_score
        return x
sample_decoder = Decoder(num_layers=3, num_heads=8, d_model=512,
                         dff=2048, vocab_size=1000, max_len=64)
output = sample_decoder(x=n,context= M)
print(output.shape)

class Transformer(keras.Model):
    def __init__(self, *, num_layers, num_heads, input_vocab_size,
                 max_len, d_model, dff, dropout_rate):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,num_heads=num_heads,
                               dff=dff, vocab_size=input_vocab_size, max_len=max_len,
                               dropout_rate=dropout_rate)
        self.final_layer = layers.Dense(input_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return  logits
transformer = Transformer(num_heads=8, num_layers=4,
                          input_vocab_size=62, max_len=64,
                          d_model=512, dff=2048,
                          dropout_rate=0.1)
model_output = transformer((M,n))
transformer.summary()

























