import os, requests
from dotenv import load_dotenv

load_dotenv()
TORCH_URL = os.getenv('TORCH_WEIGHTS_URL')
TF_URL = os.getenv('TF_WEIGHTS_URL')
DEST_DIR = os.path.dirname(__file__)

def dl(url, path):
    if not url: return False
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)
    print('Saved', path); return True

if __name__ == '__main__':
    ok1 = dl(TORCH_URL, os.path.join(DEST_DIR, 'pattern_torch.pt'))
    ok2 = dl(TF_URL, os.path.join(DEST_DIR, 'pattern_tf.h5'))
    if not (ok1 or ok2):
        print('Set TORCH_WEIGHTS_URL and/or TF_WEIGHTS_URL in .env to download pretrained weights.')
