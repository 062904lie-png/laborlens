"""Explicit deployment setup. Request handling never downloads model weights."""
from sentence_transformers import SentenceTransformer

SentenceTransformer('intfloat/multilingual-e5-base', revision='d128750597153bb5987e10b1c3493a34e5a4502a',
                    trust_remote_code=False)
print('Frozen encoder is cached. Install en_core_web_sm separately if missing.')
