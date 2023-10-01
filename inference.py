from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr_checkpoint.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False


detector = Predictor(config)

paths = glob.glob("cropWord/*.jpg")

with open('output.txt', 'w',encoding='utf-8') as f:
    outputs = []
    for path in paths:
        n = Image.open(path)
        prediction = str(detector.predict(n))
        print(prediction)

    f.write(str(outputs))




