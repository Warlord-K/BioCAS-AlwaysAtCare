from audio import AudioUtil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def preprocess(path):
    sgram = make_sgram(path)
    img_path = save_sgram(sgram)
    img = open_sgram(img_path)
    return img

def make_sgram(path,duration = 10000,sr = 8000,channel = 1,shift_pct = 0.4):
    path = str(path)
    audio_file = path
    aud = AudioUtil.open(audio_file)
    dur_aud = AudioUtil.pad_trunc(aud, duration)
    shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram

def save_sgram(sgram):
  save_path = tempfile.gettempdir() +"sgram_img"+".png"
  plt.imshow(sgram[0])
  plt.savefig(save_path)
  plt.close()
  return save_path

def open_sgram(path):
  ima = Image.open(path)
  ima_cr = ima.crop((79,155,577,344))
  ima_arr = np.array(ima_cr)
  im_n = Image.fromarray(ima_arr)
  im_n= im_n.convert("RGB")
  im_n = im_n.resize((224,224))
  ima_arr = np.array(im_n)/255
  return ima_arr