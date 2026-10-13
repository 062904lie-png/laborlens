"""Reproduce draft training data; version 2 is excluded, not fitted or scored."""
import hashlib
import json
from pathlib import Path
from expanded_examples import EXAMPLES as EXPANDED

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
 'wage_inquiry': [
  ('How is the hourly rate determined for weekend overtime?', 'Paano tinutukoy ang rate kada oras para sa overtime sa weekend?', 'Paano ginatukod ang rate kada oras para sa overtime sa weekend?'),
  ('Our boss keeps part of our earnings as a cash bond. Can we question this?', 'Kinukuha ng boss ang bahagi ng kinikita namin bilang cash bond. Maaari ba naming kuwestiyunin ito?', 'Ginakuha sang boss ang parte sang amon kinitaan bilang cash bond. Pwede bala namon ini pamangkuton?'),
  ('The work continues after the time recorded on my attendance sheet. How should this be compensated?', 'Tuloy ang trabaho pagkatapos ng oras na nakatala sa attendance sheet ko. Paano ito dapat bayaran?', 'Nagapadayon ang obra pagkatapos sang oras sa akon attendance sheet. Paano ini dapat bayran?'),
  ('I received less than the amount we agreed for the job. What can I do about my earnings?', 'Mas mababa sa napagkasunduan ang natanggap ko para sa trabaho. Ano ang magagawa ko tungkol sa kinita ko?', 'Mas manubo sa ginkasugtan ang akon nabaton para sa obra. Ano ang mahimo ko parte sa akon kinitaan?'),
 ],
 'leave_benefits': [
  ('Can unused sick days be carried over?', 'Puwede bang dalhin sa susunod ang hindi nagamit na sick days?', 'Pwede bala madala sa masunod ang wala nagamit nga sick days?'),
  ('I need time off to care for my child. Which employment benefit should I look into?', 'Kailangan kong lumiban para alagaan ang anak ko. Anong benepisyo sa trabaho ang dapat kong tingnan?', 'Kinahanglan ko magliban para atipanon ang akon bata. Ano nga benepisyo sa obra ang dapat ko tan-awon?'),
  ('How do I request a period of absence after giving birth?', 'Paano ako hihiling ng panahon ng pagliban pagkatapos manganak?', 'Paano ako magpangayo sang panahon sa pagliban pagkatapos magbata?'),
  ('Does taking a vacation use up my paid absence credits?', 'Nababawasan ba ang credits ko sa bayad na pagliban kapag nagbakasyon?', 'Mabuhinan bala ang akon credits sa bayad nga pagliban kon magbakasyon?'),
 ],
 'termination_rights': [
  ('The business closed and all staff lost their jobs. How do we check our rights?', 'Nagsara ang negosyo at nawalan kaming lahat ng trabaho. Paano namin susuriin ang mga karapatan namin?', 'Nagsira ang negosyo kag nadulaan kami tanan sang obra. Paano namon usisaon ang amon mga kinamatarong?'),
  ('My boss told me my services were no longer needed. What should I ask for in writing?', 'Sinabi ng boss na hindi na kailangan ang serbisyo ko. Ano ang dapat kong hingin na nakasulat?', 'Nagsiling ang boss nga indi na kinahanglan ang akon serbisyo. Ano ang dapat ko pangayuon nga nakasulat?'),
  ('I intend to leave the company voluntarily. How much advance notice should I give?', 'Kusa akong aalis sa kumpanya. Gaano kaaga ako dapat magbigay ng abiso?', 'Boluntaryo ako nga mahalin sa kumpanya. Daw ano ka temprano ang akon pagpahibalo?'),
  ('They removed me from the staff list without a hearing. What can I do to challenge losing my job?', 'Inalis ako sa listahan ng staff nang walang pagdinig. Paano ko makukuwestiyon ang pagkawala ng trabaho ko?', 'Ginkuha ako sa listahan sang staff nga wala pagpamati. Paano ko makuwestiyon ang pagkawala sang akon obra?'),
 ],
 '13th_month': [
  ('Which earnings are included in the thirteenth month computation?', 'Anong kinita ang kasama sa kuwenta ng thirteenth month?', 'Ano nga kinitaan ang lakip sa pagkuwenta sang thirteenth month?'),
  ('Do employees who leave midyear still receive a prorated 13th month benefit?', 'May prorated 13th month benefit pa ba ang mga umaalis sa kalagitnaan ng taon?', 'May prorated 13th month benefit pa bala ang mga nagahalin sa tunga sang tuig?'),
  ('Where is the rule about the mandatory year-end extra month payment?', 'Nasaan ang patakaran sa mandatory na dagdag na buwang bayad sa katapusan ng taon?', 'Diin ang patakaran sa mandatory nga dugang nga bulan nga bayad sa katapusan sang tuig?'),
  ('What is the difference between a discretionary bonus and the statutory 13th month?', 'Ano ang pagkakaiba ng discretionary bonus at statutory 13th month?', 'Ano ang kinalain sang discretionary bonus kag statutory 13th month?'),
 ],
 'employment_status': [
  ('How do you decide whether a person doing work is an employee or a self-employed service provider?', 'Paano malalaman kung empleyado o self-employed service provider ang isang nagtatrabaho?', 'Paano mahibaluan kon empleyado ukon self-employed service provider ang isa ka nagaobra?'),
  ('My job is tied to a harvest season. What category of worker am I?', 'Nakatali sa panahon ng anihan ang trabaho ko. Anong kategorya ako ng manggagawa?', 'Nakatali sa panahon sang ani ang akon obra. Ano ang akon kategorya bilang mamumugon?'),
  ('I am undergoing an initial trial period at work. How is my employment classified?', 'Nasa unang panahon ng pagsubok ako sa trabaho. Paano inuuri ang employment ko?', 'Ara ako sa una nga panahon sang pagtilaw sa obra. Paano ginaklasipikar ang akon employment?'),
  ('The company controls my tools, schedule, and tasks but says I run my own business. What relationship does that create?', 'Kontrolado ng kumpanya ang gamit, iskedyul, at gawain ko pero sarili ko raw ang negosyo. Anong ugnayan ang ipinapakita nito?', 'Kontrolado sang kumpanya ang gamit, iskedyul, kag buluhaton ko pero akon kuno ang negosyo. Ano nga relasyon ang ginapakita sini?'),
 ],
 'occupational_safety': [
  ('We handle corrosive chemicals with no gloves. What should be reported?', 'Humahawak kami ng nakapapasong kemikal nang walang guwantes. Ano ang dapat ireport?', 'Nagauyat kami sang makapaso nga kemikal nga wala guwantes. Ano ang dapat ireport?'),
  ('Loose scaffolding almost caused a worker to fall. How should this risk be documented?', 'Muntik nang mahulog ang manggagawa dahil sa maluwag na scaffolding. Paano itatala ang panganib?', 'Diutay na lang mahulog ang mamumugon tungod sa maluag nga scaffolding. Paano irekord ang peligro?'),
  ('The fire escape is blocked by boxes. Where should we raise the concern?', 'Nakaharang ang mga kahon sa fire escape. Saan namin idudulog ang problema?', 'Nakabara ang mga kahon sa fire escape. Diin namon idangop ang problema?'),
  ('Who must arrange protective gear and hazard training for workers?', 'Sino ang dapat maglaan ng protective gear at pagsasanay sa panganib para sa manggagawa?', 'Sin-o ang dapat maghatag sang protective gear kag paghanas sa peligro para sa mamumugon?'),
 ],
 'holiday_pay': [
  ('How does working on a legal holiday affect the amount due?', 'Paano naaapektuhan ng pagpasok sa legal holiday ang halagang matatanggap?', 'Paano makaapekto ang pag-obra sa legal holiday sa kantidad nga mabaton?'),
  ('Is a special non-working day treated differently from an ordinary workday?', 'Iba ba ang pagtrato sa special non-working day kaysa sa karaniwang araw ng trabaho?', 'Lain bala ang pagtratar sa special non-working day kaysa sa ordinaryo nga adlaw sang obra?'),
  ('What attendance conditions matter for holiday remuneration?', 'Anong kondisyon sa pagpasok ang mahalaga para sa bayad sa holiday?', 'Ano nga kondisyon sa pagsulod ang importante para sa bayad sa holiday?'),
  ('My rest day coincided with a public holiday. How do I find the applicable premium?', 'Natapat sa public holiday ang rest day ko. Paano ko hahanapin ang angkop na premium?', 'Natunong sa public holiday ang akon rest day. Paano ko pangitaon ang nagaangay nga premium?'),
 ],
 'general_inquiry': [
  ('Please explain how mediation of workplace disputes works.', 'Pakipaliwanag ang proseso ng mediation sa mga alitan sa trabaho.', 'Palihog ipaathag ang proseso sang mediation sa mga pagbininaisay sa obra.'),
  ('Thank you. Now explain the procedure for reporting discrimination.', 'Salamat. Ngayon ipaliwanag ang proseso ng pagrereklamo sa diskriminasyon.', 'Salamat. Subong ipaathag ang proseso sang pagreklamo sa diskriminasyon.'),
  ('Which documents explain how a union negotiates with management?', 'Anong dokumento ang nagpapaliwanag kung paano nakikipag-usap ang unyon sa management?', 'Ano nga dokumento ang nagapaathag kon paano nagapakighambal ang unyon sa management?'),
  ('Hello, explain the new remote attendance policy.', 'Hello, ipaliwanag ang bagong remote attendance policy.', 'Hello, ipaathag ang bag-o nga remote attendance policy.'),
 ],
 'greeting': [
  ('Hi there, assistant!', 'Magandang araw sa iyo!', 'Maayong adlaw sa imo!'),
  ('A very good morning to you.', 'Magandang umaga sa inyong lahat.', 'Maayong aga sa inyo.'),
  ('Hello and good afternoon!', 'Hello, magandang hapon po!', 'Hello, maayong hapon!'),
  ('Hey everyone!', 'Kumusta kayong lahat!', 'Maayong udto sa tanan!'),
 ],
 'small_talk': [
  ('How is everything with you?', 'Kumusta naman ang lagay mo?', 'Kumusta ang kahimtangan mo?'),
  ('I am fine. How about you?', 'Mabuti ako. Kumusta ka naman?', 'Maayo ako. Kumusta ka ya?'),
  ('Has it been a pleasant day for you?', 'Naging maganda ba ang araw mo?', 'Nangin maayo bala ang adlaw mo?'),
  ('What have you been up to?', 'Ano ang pinagkakaabalahan mo?', 'Ano ang imo ginahimo subong?'),
 ],
 'identity': [
  ('Tell me your name, please.', 'Sabihin mo ang pangalan mo.', 'Ihambal ang imo ngalan.'),
  ('Am I chatting with software or a real person?', 'Software ba o totoong tao ang kausap ko?', 'Software bala ukon matuod nga tawo ang akon kaistorya?'),
  ('Give a brief introduction of yourself.', 'Magbigay ka ng maikling pagpapakilala.', 'Maghatag sang malip-ot nga pagpakilala sa imo kaugalingon.'),
  ('Who is the assistant answering me?', 'Sino ang assistant na sumasagot sa akin?', 'Sin-o ang assistant nga nagasabat sa akon?'),
 ],
 'capability_question': [
  ('Tell me about the services you offer.', 'Sabihin mo kung anong serbisyo ang maibibigay mo.', 'Ihambal kon ano nga serbisyo ang imo mahatag.'),
  ('What subjects are you able to discuss?', 'Anong paksa ang kaya mong talakayin?', 'Ano nga mga topiko ang imo masarangan istoryahan?'),
  ('Can I ask questions in Filipino?', 'Puwede ba akong magtanong sa Filipino?', 'Pwede bala ako makapamangkot sa Filipino?'),
  ('Describe the limits of the help you provide.', 'Ilarawan mo ang hangganan ng tulong na ibinibigay mo.', 'Ilaragway ang limitasyon sang bulig nga imo ginahatag.'),
 ],
 'gratitude': [
  ('Your answer really helped me.', 'Nakatulong talaga sa akin ang sagot mo.', 'Nakabulig gid sa akon ang imo sabat.'),
  ('I am grateful for your assistance.', 'Nagpapasalamat ako sa tulong mo.', 'Nagapasalamat ako sa imo bulig.'),
  ('Thank you so much for explaining!', 'Salamat nang marami sa pagpapaliwanag!', 'Salamat gid sa pagpaathag!'),
  ('You cleared up my confusion. Thanks a lot.', 'Nalinawan na ako. Salamat nang marami.', 'Nasanagan na ako. Madamo nga salamat.'),
 ],
 'farewell': [
  ('I have to go now. Talk to you soon.', 'Aalis na ako. Usap tayo ulit.', 'Mahalin na ako. Istorya kita liwat.'),
  ('Until next time, goodbye.', 'Hanggang sa muli, paalam.', 'Asta sa liwat, halong.'),
  ('That is all for now. Have a nice evening.', 'Iyon lang muna. Magandang gabi sa iyo.', 'Amo lang anay. Maayong gab-i sa imo.'),
  ('Signing off now. See you soon!', 'Magpapaalam na ako. Kita tayo ulit!', 'Mapaalam na ako. Kitaay kita liwat!'),
 ],
}


def main():
    original = ROOT / 'deliverables/nlp-dataset/laborlens_nlp_all_150_draft.json'
    data = json.loads(original.read_text(encoding='utf-8'))
    rows = [{'text':case['question'],'intent':case['expected_intent'],'language':case['expected_language'],
             'group':case['id'].rsplit('_',1)[0], 'source':'v1_draft'} for case in data['cases']]
    for intent, triples in EXAMPLES.items():
        for index, triple in enumerate(triples):
            for language, text in zip(('en','fil','hil'),triple):
                rows.append({'text':text,'intent':intent,'language':language,'group':f'train_{intent}_{index}', 'source':'training_draft'})
    for intent, triples in EXPANDED.items():
        for index, triple in enumerate(triples):
            for language, text in zip(('en','fil','hil'), triple):
                rows.append({'text': text, 'intent': intent, 'language': language,
                             'group': f'expanded_{intent}_{index}', 'source': 'expanded_training_draft'})
    # This check reads only question strings to prevent exact leakage; no v2
    # expected label is read, used for fitting, or used to select parameters.
    reserved = json.loads((ROOT / 'deliverables/nlp-challenge-v2/laborlens_nlp_challenge_v2_126_draft.json').read_text(encoding='utf-8'))
    normalize = lambda text: ' '.join(text.casefold().split())
    if {normalize(row['text']) for row in rows} & {normalize(case['question']) for case in reserved['cases']}:
        raise ValueError('Training corpus overlaps reserved v2 questions')
    reserved_v3 = json.loads((ROOT / 'deliverables/nlp-challenge-v3/laborlens_nlp_challenge_v3_168_draft.json').read_text(encoding='utf-8'))
    if {normalize(row['text']) for row in rows} & {normalize(case['question']) for case in reserved_v3['cases']}:
        raise ValueError('Training corpus overlaps reserved v3 questions')
    path = Path(__file__).with_name('nlp_training_draft.json')
    path.write_text(json.dumps({'status':'synthetic_unreviewed','rows':rows,
                               'v1_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                               'reserved_v2_used_for_training':False,
                               'reserved_v3_used_for_training':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(rows)} training examples to {path}')


if __name__ == '__main__':
    main()
