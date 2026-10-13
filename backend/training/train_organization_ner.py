"""Train a small multilingual ORG recognizer from separate synthetic spans."""
import json
import random
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch
from expanded_examples import EXAMPLES

ROOT = Path(__file__).resolve().parents[2]
NAMES = [
 'Dalisay Logistics','Bagwis Foods','Tala Digital','Sinag Manufacturing',
 'Lakbay Trading','Likas Corporation','Mithi Services','Kawayan Bank',
 'Sulo Hospital','Liwanag Foundation','Habagat Industries','Salinlahi University',
 'Green Valley Transport','North Shore Systems','Clearwater Retail','Silver Lake Media',
 'Blue Harbor Packaging','Cedar Grove Engineering','East River Supplies','Bright Field Textiles',
 'Mariposa','Soluna','Pintado','Lumina',
]
NAMES += ['Amihan', 'Luntian', 'Hiraya', 'Alon', 'Mayumi', 'Silangan', 'Mabini', 'Balangay',
          'Harbor Peak Repair Center', 'Willow Gate Service Hub', 'River Stone Cargo Works',
          'Maple Crest Textile Group', 'Golden Reed Analytics', 'South Ridge Care Institute',
          'Evergreen Signal Systems', 'Coral Bay Equipment Cooperative']
TEMPLATES = [
 'I work at {name}. Can you explain my employment rights?',
 'The notice was issued by {name}. What should I check?',
 '{name} handles our payroll and withheld my earnings.',
 'My maternity application was sent to {name}.',
 'A manager from {name} told us the site will close.',
 'The document names {name} as the employer.',
 'Nagtatrabaho ako sa {name}. Ano ang karapatan ko?',
 'Galing sa {name} ang abiso. Ano ang dapat kong suriin?',
 '{name} ang nangangasiwa sa sahod namin.',
 'Ipinadala ko sa {name} ang aplikasyon sa leave.',
 'Sinabi ng manager ng {name} na magsasara ang pasilidad.',
 'Nakalagay sa dokumento na {name} ang employer.',
 'Nagaobra ako sa {name}. Ano ang akon kinamatarong?',
 'Halin sa {name} ang abiso. Ano ang dapat ko usisaon?',
 '{name} ang nagadumala sang amon suhol.',
 'Ginpadala ko sa {name} ang aplikasyon sa leave.',
 'Nagsiling ang manager sang {name} nga magasira ang pasilidad.',
 'Nakasulat sa dokumento nga {name} ang employer.',
]
TEMPLATES += [
 'At {name}, my paid vacation balance was reduced. Which records should I request?',
 '{name} received the request on 2028-02-19 and will send a reply.',
 'My contract with {name} lists an allowance of PHP 420.',
 'I sent my resignation to {name} after completing the notice period.',
 'The holiday payroll of {name} has a missing premium.',
 'May aplikasyon ako sa {name} na may petsang 2028-11-06.',
 'Hindi pa inilalabas ng {name} ang bayad sa dagdag na oras.',
 'Nasa rekord ng {name} ang leave credits ko.',
 'Sinuri ko ang kasunduan sa {name} at ang regularization date.',
 'Mula sa {name} ang memo tungkol sa kaligtasan.',
 'May aplikasyon ako sa {name} nga may petsa 2028-11-06.',
 'Wala pa ginahatag sang {name} ang bayad sa dugang nga oras.',
 'Ara sa rekord sang {name} ang akon leave credits.',
 'Ginusisa ko ang kasugtanan sa {name} kag ang regularization date.',
 'Halin sa {name} ang memo parte sa kaluwasan.',
]
NEGATIVES = [
 'Hello, LaborLens. Can you explain your capabilities?',
 'Can you understand English, Filipino, and Hiligaynon?',
 'The document is titled "Good Morning Handbook". Explain it.',
 'PHP 500 was recorded but I received PHP 450.',
 'I need to understand R.A. 11058 and P.D. 851.',
 'Salamat sa bulig. Paano ko masusi ang payroll?',
 'Nakatanggap ako ng notice mula sa boss ko.',
 'Nagtrabaho ako pero hindi ako binayaran.',
 'Nakabaton ako sang abiso halin sa akon amo.',
 'Wala pa nabayaran ang akon suhol.',
 'That is all, thank you and goodbye.',
 'The payroll says 900 pesos and the leave form says 2027-04-18.',
]


def main():
    fix_random_seed(42)
    rows = []
    for name in NAMES:
        for template in TEMPLATES:
            text = template.format(name=name)
            start = text.index(name)
            rows.append({'text':text, 'entities':[(start,start+len(name),'ORG')]})
    original = json.loads((ROOT / 'deliverables/nlp-dataset/laborlens_nlp_all_150_draft.json').read_text(encoding='utf-8'))
    negatives = NEGATIVES + [case['question'] for case in original['cases'] if not case['expected_entities'].get('organizations')]
    negatives += [text for triples in EXAMPLES.values() for triple in triples for text in triple]
    # Legal titles, dates, language names, and greeting text are not ORGs.
    for title in ('Presidential Decree 442', 'Republic Act 11313', 'Article 95', 'R.A. 10911'):
        negatives += [f'Please find {title} in the knowledge base.',
                      f'Pakihanap ang {title} sa knowledge base.',
                      f'Palihog pangitaa ang {title} sa knowledge base.']
    for calendar in ('2028-07-11', '22 March 2028', 'March 22, 2028'):
        negatives += [f'The contract starts on {calendar}. What records matter?',
                      f'Nagsimula ang kontrata noong {calendar}. Ano ang dapat suriin?',
                      f'Nagsugod ang kontrata sang {calendar}. Ano ang dapat usisaon?']
    rows += [{'text':text,'entities':[]} for text in negatives]
    for name in ('HR', 'Human Resources', 'Payroll Department', 'Finance Division'):
        for template in ('Please submit the application to {name}.',
                         'Ipinasa ko ang form sa {name}.',
                         'Ginpadala ko ang form sa {name}.'):
            text = template.format(name=name)
            rows.append({'text':text, 'entities':[(text.index(name),text.index(name)+len(name),'ORG')]})
    # A second organization in the same sentence is independently annotated.
    for first,second in zip(NAMES[::2],NAMES[1::2]):
        for template in ('I moved from {a} to {b}.','Lumipat ako mula sa {a} patungo sa {b}.','Nagsaylo ako halin sa {a} pakadto sa {b}.'):
            text = template.format(a=first,b=second)
            rows.append({'text':text,'entities':[(text.index(name),text.index(name)+len(name),'ORG') for name in (first,second)]})
    path = Path(__file__).with_name('organization_training_draft.json')
    # Questions only: ensure no evaluated v2/v3 sentence enters NER fitting.
    normalize = lambda text: ' '.join(text.casefold().split())
    for version, count in (('v2',126),('v3',168)):
        reserved = json.loads((ROOT / f'deliverables/nlp-challenge-{version}/laborlens_nlp_challenge_{version}_{count}_draft.json').read_text(encoding='utf-8'))
        if {normalize(row['text']) for row in rows} & {normalize(case['question']) for case in reserved['cases']}:
            raise ValueError(f'Organization corpus overlaps {version}')
    path.write_text(json.dumps({'status':'synthetic_unreviewed','rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    nlp = spacy.blank('xx')
    ner = nlp.add_pipe('ner')
    ner.add_label('ORG')
    examples = [Example.from_dict(nlp.make_doc(row['text']),{'entities':row['entities']}) for row in rows]
    optimizer = nlp.initialize(lambda: examples)
    rng = random.Random(42)
    for epoch in range(20):
        rng.shuffle(examples)
        losses = {}
        for batch in minibatch(examples,size=32):
            nlp.update(batch,sgd=optimizer,drop=0.2,losses=losses)
        print(f'Organization NER epoch {epoch+1}: {losses}',flush=True)
    nlp.meta.update(name='laborlens_multilingual_org_draft',version='0.2.0',description='Trained synthetic draft; requires independent review and evaluation.')
    out = ROOT / 'backend/models/nlp_classifier/organization_ner'
    out.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out)
    print(f'Saved {out} from {len(rows)} labeled sentences.')


if __name__ == '__main__':
    main()
