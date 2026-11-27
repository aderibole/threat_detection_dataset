# import os, cv2
# from tqdm import tqdm
# import pandas as pd
# from PIL import Image
# import csv
# import requests
# from transformers import BlipProcessor, BlipForConditionalGeneration

# video_dir = "threat_detection_dataset/UCF_Crimes/Videos/Stealing"
# frame_dir = "threat_detection_dataset/frames"
# os.makedirs(frame_dir, exist_ok=True)

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def extract_frames(every_n_frames=30, max_frames_per_video=100):
#     with open('threat_detection_dataset/annotations.csv', 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for video_file in tqdm(os.listdir(video_dir)):
#             if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
#                 continue
#             video_path = os.path.join(video_dir, video_file)
#             cap = cv2.VideoCapture(video_path)
#             base = os.path.splitext(video_file)[0]
#             i, saved = 0, 0
#             while cap.isOpened() and saved < max_frames_per_video:
#                 ret, frame = cap.read()
#                 if not ret: break
#                 if i % every_n_frames == 0:
#                     frame_name = f"{base}_frame_{i:05d}.jpg"
#                     cv2.imwrite(os.path.join(frame_dir, frame_name), frame)
#                     image = Image.open(f"threat_detection_dataset/frames/{frame_name}").convert('RGB')

#                     inputs = processor(images=image, return_tensors="pt")
#                     out = model.generate(**inputs)
#                     caption = processor.decode(out[0], skip_special_tokens=True)
#                     print(f"{frame_dir}/{frame_name}", caption)
#                     writer.writerow([frame_name, caption])
#                     saved += 1
#                 i += 1
#             cap.release()

# extract_frames()

import tensorflow as tf
import pandas as pd
import os

img_root = "threat_detection_dataset/frames"
csv_file = "threat_detection_dataset/annotations.csv"
tfrecord_path = "threat_detection_dataset/tfrecords/threat_captioning.tfrecord"
os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)

def bytes_feature(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def build_tfrecord():
    df = pd.read_csv(csv_file)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for _, row in df.iterrows():
            img_path = os.path.join(img_root, row["filename"])
            if not os.path.exists(img_path): continue
            caption = row["caption"]
            img_bytes = open(img_path, "rb").read()
            feature = tf.train.Features(feature={
                "image": bytes_feature(img_bytes),
                "caption": bytes_feature(caption.encode("utf-8"))
            })
            # print(tf.train.Example(features=feature).SerializeToString())
            writer.write(tf.train.Example(features=feature).SerializeToString())
    print("✅ TFRecord created.")

build_tfrecord()

import numpy as np
import re, json, cv2
from collections import Counter
from tqdm import tqdm

VOCAB_PATH      = "threat_detection_dataset/vocab.json"
TFRECORD_PATH = "threat_detection_dataset/tfrecords/threat_captioning.tfrecord"
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS          = 200
EMB_DIM         = 256
LSTM_UNITS      = 512
MIN_FREQ        = 3          # drop rare tokens
THREAT_WORDS    = {"gun","knife","fire","fight","accident","kidnap"}

def build_or_load_vocab(tfrecord_path, vocab_path=VOCAB_PATH, min_freq=MIN_FREQ):
    if os.path.exists(vocab_path):
        return json.load(open(vocab_path))
    
    # collect captions
    counter = Counter()
    for raw in tf.data.TFRecordDataset(tfrecord_path):
        caption = tf.io.parse_single_example(raw,
                 {"caption": tf.io.FixedLenFeature([], tf.string)})["caption"].numpy().decode()
        counter.update(re.findall(r"\w+|[^\w\s]", caption.lower()))
    
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + \
            [w for w, f in counter.items() if f >= min_freq]
    w2i = {w:i for i,w in enumerate(vocab)}
    i2w = {i:w for w,i in w2i.items()}
    json.dump({'w2i':w2i,'i2w':i2w}, open(vocab_path,"w"))
    return {'w2i':w2i,'i2w':i2w}

vocab = build_or_load_vocab(TFRECORD_PATH)
w2i, i2w, VOCAB_SIZE = vocab['w2i'], vocab['i2w'], len(vocab['w2i'])

def encode_caption(caption):
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8")
    toks = re.findall(r"\w+|[^\w\s]", caption.lower())
    ids  = [w2i.get(t, w2i['<unk>']) for t in toks]
    return np.array([w2i['<bos>']] + ids + [w2i['<eos>']], dtype=np.int32)
    # return [w2i['<bos>']] + ids + [w2i['<eos>']]

def parse_tfrecord(example):
    feat = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "caption": tf.io.FixedLenFeature([], tf.string)
    }
    x = tf.io.parse_single_example(example, feat)
    img = tf.image.decode_jpeg(x["image"], channels=3)

    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    diff = tf.abs(h - w)
    pad_top = pad_bottom = pad_left = pad_right = 0

    def pad_height():
        pad = diff // 2
        return pad, diff - pad, 0, 0

    def pad_width():
        pad = diff // 2
        return 0, 0, pad, diff - pad

    pad_top, pad_bottom, pad_left, pad_right = tf.cond(
        h > w,
        true_fn=pad_width,
        false_fn=pad_height
    )

    img = tf.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)

    img = tf.image.resize(img, IMG_SIZE) / 255.0

    cap = tf.numpy_function(encode_caption, [x["caption"]], tf.int32)
    cap.set_shape([None])
    return img, cap

ds = tf.data.TFRecordDataset(TFRECORD_PATH)\
       .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

def pad_batch(images, captions):
    def _pad(caps):
        return tf.keras.preprocessing.sequence.pad_sequences(caps, padding='post', value=w2i['<pad>'])
    
    captions = tf.numpy_function(_pad, [captions], tf.int32)
    captions.set_shape([None, None])
    return images, captions

# train_ds = ds.shuffle(2048).padded_batch(BATCH_SIZE, padded_shapes=([None,None,3],[None]))\
#               .map(pad_batch).prefetch(tf.data.AUTOTUNE)
train_ds = ds.shuffle(2048).padded_batch(
    BATCH_SIZE,
    padded_shapes=([None, None, 3], [None]),
    padding_values=(0.0, w2i['<pad>'])
).prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, Model

def build_encoder():
    inputs = layers.Input(shape=(*IMG_SIZE,3))
    x = inputs
    for filters in [32,64,128,256]:
        x = layers.Conv2D(filters,3,padding='same',activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)          # /2 each block
    # Output 7×7×256 ⇒ flatten spatial dims
    x = layers.Reshape((-1,256))(x)          # (B,49,256)
    return Model(inputs,x,name="Encoder")

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, units, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.units = units

        self.embedding = layers.Embedding(vocab_size, emb_dim)
        self.attn_dense = layers.Dense(units)
        self.U = layers.Dense(units)
        self.W = layers.Dense(1)
        self.lstm = layers.LSTM(units, return_state=True, return_sequences=True)
        self.fc   = layers.Dense(vocab_size)

    def call(self, features, seq):
        # batch_size = tf.shape(features)[0]
        # T = tf.shape(seq)[1]

        # hidden_state = tf.zeros((batch_size, self.units))
        # cell_state   = tf.zeros_like(hidden_state)
        # emb = self.embedding(seq)

        # outputs = tf.TensorArray(tf.float32, size=T)

        # def condition(t, *_):
        #     return t < T

        # def body(t, outputs, hidden_state, cell_state):
        #     context = self._attention(features, hidden_state)
        #     x = tf.concat([tf.expand_dims(context, 1), emb[:, t:t+1, :]], axis=-1)
        #     out, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])
        #     out = tf.squeeze(out, axis=1)
        #     outputs = outputs.write(t, out)
        #     return t + 1, outputs, hidden_state, cell_state

        # t0 = tf.constant(0)
        # _, outputs, _, _ = tf.while_loop(
        #     condition,
        #     body,
        #     loop_vars=[t0, outputs, hidden_state, cell_state]
        # )

        # outputs = outputs.stack()               # (T, B, units)
        # outputs = tf.transpose(outputs, [1, 0, 2])  # (B, T, units)
        # logits  = self.fc(outputs)              # (B, T, vocab_size)
        # return logits

        # Attention context (average as static, or learn with another layer)
        context = tf.reduce_mean(features, axis=1, keepdims=True)  # (B, 1, 256)

        emb = self.embedding(seq)  # (B, T, E)
        context_repeated = tf.repeat(context, repeats=tf.shape(seq)[1], axis=1)  # (B, T, 256)

        x = tf.concat([context_repeated, emb], axis=-1)  # (B, T, 256 + E)

        lstm_out, _, _ = self.lstm(x)  # (B, T, units)
        logits = self.fc(lstm_out)    # (B, T, vocab_size)
        return logits

        features: (B, 49, 256), seq: (B, T)
        hidden_state = tf.zeros((tf.shape(features)[0], self.lstm.units))
        cell_state   = tf.zeros_like(hidden_state)
        outputs = []
        emb = self.embedding(seq)                         # (B, T, E)

        T = tf.shape(seq)[1]
        for t in tf.range(T):                   # teacher forcing
            context = self._attention(features, hidden_state)  # (B,256)
            x = tf.concat([tf.expand_dims(context,1), emb[:,t:t+1,:]], axis=-1)
            out, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])
            outputs.append(out)
        outputs = tf.concat(outputs, axis=1)              # (B, T-1, units)
        logits  = self.fc(outputs)
        return logits

    def _attention(self, features, hidden):
        # Bahdanau
        score = tf.nn.tanh(self.attn_dense(features) + tf.expand_dims(self.U(hidden),1))
        alpha = tf.nn.softmax(self.W(score), axis=1)      # (B, 49, 1)
        context = tf.reduce_sum(alpha*features, axis=1)   # (B, 256)
        return context

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "units": self.units
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

encoder = build_encoder()
decoder = Decoder(VOCAB_SIZE, EMB_DIM, LSTM_UNITS)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),  # images: (B, H, W, 3)
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)           # captions: (B, T)
])
def train_step(imgs, caps):
    dec_inp  = caps[:, :-1]
    dec_targ = caps[:, 1:]
    with tf.GradientTape() as tape:
        feats  = encoder(imgs, training=True)
        logits = decoder(feats, dec_inp, training=True)
        mask = tf.cast(dec_targ != w2i['<pad>'], tf.float32)
        loss = loss_obj(dec_targ, logits)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    grads = tape.gradient(loss, encoder.trainable_weights + decoder.trainable_weights)
    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
    return loss

save_dir = os.path.join('threat_detection_system', 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
encoder_filepath = os.path.join(save_dir, "threat_detection_system_encoder_model.keras")
decoder_filepath = os.path.join(save_dir, "threat_detection_system_decoder_model.keras")

best_loss = float("inf")
for epoch in range(EPOCHS):
    epoch_loss = []
    for imgs, caps in tqdm(train_ds):
        loss = train_step(imgs, caps)
        epoch_loss.append(loss.numpy())
    
    avg_loss = np.mean(epoch_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        encoder.save(encoder_filepath)
        decoder.save(decoder_filepath)
