"""Fit frozen-embedding logistic regression and a character language model."""
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('nlp_training_draft.json')
OUT = ROOT / 'backend/models/nlp_classifier'
ENCODER = 'intfloat/multilingual-e5-base'
REVISION = 'd128750597153bb5987e10b1c3493a34e5a4502a'


def language_model():
    return make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(2,5), sublinear_tf=True, min_df=1),
                         LogisticRegression(C=10, max_iter=1000, class_weight='balanced', random_state=42))


def intent_model():
    return LogisticRegression(C=10, max_iter=1000, class_weight='balanced', random_state=42)


def main():
    torch.set_num_threads(4)
    payload = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = payload['rows']
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(ENCODER, revision=REVISION, local_files_only=True, trust_remote_code=False, device='cpu')
    texts = [row['text'] for row in rows]
    vectors = encoder.encode(['query: ' + text for text in texts], normalize_embeddings=True,
                             batch_size=16, show_progress_bar=True)
    intents = [row['intent'] for row in rows]
    languages = [row['language'] for row in rows]
    groups = defaultdict(set)
    for row in rows:
        groups[row['intent']].add(row['group'])
    reserved = set()
    rng = random.Random(42)
    for names in groups.values():
        names = sorted(names)
        rng.shuffle(names)
        reserved.update(names[:max(1,round(len(names)*0.2))])
    train = [i for i,row in enumerate(rows) if row['group'] not in reserved]
    validation = [i for i,row in enumerate(rows) if row['group'] in reserved]
    classifier = intent_model().fit(vectors[train], [intents[i] for i in train])
    language = language_model().fit([texts[i] for i in train], [languages[i] for i in train])
    validation_scores = {}
    for name, expected, predicted in (
        ('intent',[intents[i] for i in validation],classifier.predict(vectors[validation])),
        ('language',[languages[i] for i in validation],language.predict([texts[i] for i in validation]))):
        validation_scores[name] = {'accuracy':accuracy_score(expected,predicted), 'macro_f1':f1_score(expected,predicted,average='macro',zero_division=0)}
    # Freeze fixed parameters and refit on training corpus only. V2 is neither
    # loaded nor scored here, and never influences model/threshold selection.
    conversational = {'greeting','small_talk','identity','capability_question','gratitude','farewell'}
    routes = ['conversation' if intent in conversational else 'content' for intent in intents]
    route = intent_model().fit(vectors[train], [routes[i] for i in train])
    validation_scores['route'] = {'accuracy': accuracy_score([routes[i] for i in validation], route.predict(vectors[validation]))}
    models = {'intent':intent_model().fit(vectors,intents), 'language':language_model().fit(texts,languages),
              'route':intent_model().fit(vectors,routes),
              'conversation_labels':{'greeting','small_talk','identity','capability_question','gratitude','farewell'}}
    path = OUT / 'classifiers.joblib'
    joblib.dump(models,path)
    org_path = OUT / 'organization_ner'
    if not (org_path / 'config.cfg').is_file():
        raise RuntimeError('Run train_organization_ner.py before packaging classifiers.')
    manifest = {'model_version':'learned-multilingual-2', 'encoder':ENCODER,'encoder_revision':REVISION,
                'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'training_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'training_status':payload['status'], 'training_count':len(rows),
                'training_groups':len({row['group'] for row in rows}), 'v2_used_for_training':False,
                'v3_used_for_training':False, 'routing_method':'learned_binary_content_conversation',
                'sklearn_version':sklearn.__version__, 'created_at':datetime.now(timezone.utc).isoformat(),
                'validation_count':len(validation),'validation_groups':sorted(reserved),'validation':validation_scores,
                'organization_training_sha256':hashlib.sha256(Path(__file__).with_name('organization_training_draft.json').read_bytes()).hexdigest(),
                'organization_artifacts':{str(path.relative_to(org_path)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in org_path.rglob('*') if path.is_file()},
                'source_sha256':{name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                 ('backend/training/train_nlp_classifier.py','backend/training/train_organization_ner.py','backend/routers/services/learned_nlp.py')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
