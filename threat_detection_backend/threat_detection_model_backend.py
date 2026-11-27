from tensorflow.keras.models import load_model
import tensorflow as tf
import json, os, cv2
from tensorflow.keras import layers
from PIL import Image
import numpy as np
from tqdm import tqdm

EMB_DIM         = 256
LSTM_UNITS      = 512
VOCAB_PATH      = "C:/Users/jlrab/threat_detection_dataset/vocab.json"

def build_or_load_vocab(vocab_path=VOCAB_PATH):
    if os.path.exists(vocab_path):
        return json.load(open(vocab_path))

vocab = build_or_load_vocab()
w2i, i2w, VOCAB_SIZE = vocab['w2i'], vocab['i2w'], len(vocab['w2i'])

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
        hidden_state = tf.zeros((tf.shape(features)[0], self.lstm.units))
        cell_state   = tf.zeros_like(hidden_state)
        outputs = []
        emb = self.embedding(seq)                         

        for t in range(seq.shape[1]):                   
            context = self._attention(features, hidden_state)  
            x = tf.concat([tf.expand_dims(context,1), emb[:,t:t+1,:]], axis=-1)
            out, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])
            outputs.append(out)
        outputs = tf.concat(outputs, axis=1)              
        logits  = self.fc(outputs)
        return logits

    def _attention(self, features, hidden):
        score = tf.nn.tanh(self.attn_dense(features) + tf.expand_dims(self.U(hidden),1))
        alpha = tf.nn.softmax(self.W(score), axis=1)      
        context = tf.reduce_sum(alpha*features, axis=1)   
        return context

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "units": self.units
        }

    def build_from_config(self, config):
        dummy_features = tf.zeros((1, 196, 256)) 
        dummy_seq = tf.zeros((1, 10), dtype=tf.int32)  # (B, caption length)
        self(dummy_features, dummy_seq)
        return self

    @classmethod
    def from_config(cls, config):
        return cls(**config)

save_dir = os.path.join('C:/Users/jlrab/threat_detection_system', 'saved_modelsLast')
encoder_filepath = os.path.join(save_dir, "threat_detection_system_encoder_model.keras")
decoder_filepath = os.path.join(save_dir, "threat_detection_system_decoder_model.keras")

encoder = load_model(encoder_filepath)
decoder = load_model(decoder_filepath, custom_objects={"Decoder": Decoder})

def greedy_caption(img):
    feat = encoder(tf.expand_dims(img, 0), training=False)
    seq = [w2i['<bos>']]
    for _ in range(20):
        logits = decoder(feat, tf.constant([seq]), training=False)
        next_id = int(tf.argmax(logits[0, -1]).numpy())
        if next_id == w2i['<eos>']:
            break
        seq.append(next_id)
    sentence = ""
    for i in seq[1:]:
        word = i2w[str(i)]
        print(i, word)
        sentence = sentence + " " + word
    return sentence

    # return " ".join([i2w[str(i)] for i in seq[1:]])

# image = cv2.imread("C:/Users/jlrab/OneDrive/Documents/test_model/Shoplifting007_x264_frame_02400.jpg", cv2.IMREAD_UNCHANGED)

# def process_image_folder():
#     class_path = "C:/Users/jlrab/OneDrive/Documents/test_model"
#     for file in tqdm(os.listdir(class_path), desc=f"Loading"):
#         img_path = os.path.join(class_path, file)
#         print(file)
#         image = np.array(Image.open(img_path).convert('RGB'))
#         resized = cv2.resize(image, (224, 224)) / 255.0
#         caption = greedy_caption(resized)
#         print(caption)  

# process_image_folder()