"""Learned intent and language models. No phrase lists or evaluation lookup."""
import hashlib
import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from collections import OrderedDict

import numpy as np

ARTIFACT_DIR = Path(os.environ.get('LABORLENS_NLP_MODEL_DIR', Path(__file__).resolve().parents[2] / 'models/nlp_classifier'))
_lock = threading.RLock()
_bundle = None
_encoder = None
_organization_ner = None
_scores = OrderedDict()


def load_bundle():
    global _bundle
    with _lock:
        if _bundle is None:
            import joblib
            import sklearn
            manifest = json.loads((ARTIFACT_DIR / 'manifest.json').read_text(encoding='utf-8'))
            if sklearn.__version__ != manifest['sklearn_version']:
                raise RuntimeError('Install the pinned scikit-learn version before loading the NLP classifier.')
            path = ARTIFACT_DIR / 'classifiers.joblib'
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['artifact_sha256']:
                raise RuntimeError('NLP model checksum mismatch. Rebuild the classifier artifacts.')
            # Only this server-owned artifact is deserialized, never uploads.
            _bundle = (joblib.load(path), manifest)
        return _bundle


def get_encoder():
    global _encoder
    with _lock:
        _, manifest = load_bundle()
        if _encoder is None:
            chroma = sys.modules.get('routers.services.chroma_service')
            shared = getattr(chroma, '_embedder', None)
            if shared is not None:
                config = getattr(getattr(shared[0], 'auto_model', None), 'config', None)
                if getattr(config, '_commit_hash', None) == manifest['encoder_revision']:
                    _encoder = shared
            if _encoder is None:
                from sentence_transformers import SentenceTransformer
                # Startup/inference never downloads or trains a model.
                _encoder = SentenceTransformer(manifest['encoder'], revision=manifest['encoder_revision'],
                                               local_files_only=True, trust_remote_code=False)
        return _encoder


def encode(texts):
    with _lock:
        return get_encoder().encode(['query: ' + text for text in texts], normalize_embeddings=True,
                                    batch_size=16, show_progress_bar=False)


def prime(texts):
    """Batch inference for admin evaluations; never fits or reads labels."""
    models, _ = load_bundle()
    with _lock:
        missing = list(dict.fromkeys(text for text in texts if text not in _scores))
        if missing:
            vectors = encode(missing)
            probabilities = models['intent'].predict_proba(vectors)
            routes = models['route'].predict(vectors) if 'route' in models else [None] * len(missing)
            for text, scores, route in zip(missing, probabilities, routes):
                _scores[text] = (scores, route)
            while len(_scores) > 256:
                _scores.popitem(last=False)


@lru_cache(maxsize=256)
def detect_language(text):
    models, _ = load_bundle()
    return str(models['language'].predict([text])[0])


@lru_cache(maxsize=256)
def organizations(text):
    global _organization_ner
    with _lock:
        _, manifest = load_bundle()
        if _organization_ner is None:
            import spacy
            for name, expected in manifest['organization_artifacts'].items():
                if hashlib.sha256((ARTIFACT_DIR / 'organization_ner' / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Organization model checksum mismatch.')
            _organization_ner = spacy.load(ARTIFACT_DIR / 'organization_ner')
        return tuple(ent.text for ent in _organization_ner(text).ents if ent.label_ == 'ORG')


@lru_cache(maxsize=256)
def classify(text):
    models, _ = load_bundle()
    with _lock:
        prime([text])
        scores, route = _scores[text]
        _scores.move_to_end(text)
    order = np.argsort(scores)[::-1]
    candidate = str(models['intent'].classes_[order[0]])
    score = float(scores[order[0]])
    margin = float(scores[order[0]] - scores[order[1]])
    # A separately learned content/conversation decision protects retrieval.
    # Low multiclass scores alone are not evidence of substantive content.
    ordinary = models['conversation_labels']
    uncertain = score < 0.40 or margin < 0.10
    if route is None:  # Compatibility with the original deployed artifact.
        routed = 'general_inquiry' if candidate in ordinary and uncertain else candidate
    else:
        routed = 'general_inquiry' if candidate in ordinary and route == 'content' else candidate
    return {'intent': routed, 'candidate': candidate, 'score': score, 'margin': margin,
            'uncertain': uncertain, 'route': route, 'language': detect_language(text),
            'method': 'learned_multilingual_v2' if route is not None else 'learned_multilingual_v1'}


def metadata():
    _, manifest = load_bundle()
    return {key: manifest[key] for key in ('model_version','encoder','encoder_revision','artifact_sha256',
                                          'training_sha256','training_status','training_count','source_sha256',
                                          'organization_training_sha256','organization_artifacts')}
