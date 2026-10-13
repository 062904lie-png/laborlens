"""Separately authored training examples; never used as an evaluation set.

Labels are synthetic and require fluent human review. Scenario translations
share a group so they cannot cross the internal validation boundary.
"""
EXAMPLES = {
 'wage_inquiry': [
 ('The payslip omits two hours I worked. How do I request a correction?', 'Wala sa payslip ang dalawang oras na ipinagtrabaho ko. Paano ako hihiling ng pagwawasto?', 'Wala sa payslip ang duha ka oras nga gin-obra ko. Paano ako magpangayo sang pagtadlong?'),
 ('Our wages arrive late every month. What payment schedule should be followed?', 'Buwan-buwan nahuhuli ang sahod namin. Anong iskedyul ng pagbabayad ang dapat sundin?', 'Kada bulan nalangan ang amon suhol. Ano nga iskedyul sang pagbayad ang dapat sundon?'),
 ('Hello, can you help me check the overtime rate on this payroll?', 'Hello, matutulungan mo ba akong suriin ang overtime rate sa payroll?', 'Hello, mabuligan mo bala ako mag-usisa sang overtime rate sa payroll?')],
 'leave_benefits': [
 ('May the employer replace my unused vacation credits with money?', 'Maaari bang palitan ng employer ng pera ang hindi ko nagamit na vacation credits?', 'Mahimo bala baylohan sang employer sang kwarta ang akon wala nagamit nga vacation credits?'),
 ('My wife gave birth and I want to apply for paternity leave.', 'Nanganak ang asawa ko at nais kong mag-apply ng paternity leave.', 'Nagbata ang akon asawa kag gusto ko mag-apply sang paternity leave.'),
 ('I was absent due to illness. Can my available sick leave cover those days?', 'Lumiban ako dahil may sakit. Maaari bang gamitin ang natitira kong sick leave sa mga araw na iyon?', 'Nagliban ako tungod sa balatian. Pwede bala gamiton ang akon nabilin nga sick leave sa sina nga mga adlaw?')],
 'termination_rights': [
 ('How much notice is needed before I voluntarily resign?', 'Gaano katagal ang abiso bago ako kusang magbitiw?', 'Daw ano kalawig ang abiso antes ako boluntaryo nga maghalin sa obra?'),
 ('I lost my position during downsizing. What separation documents are necessary?', 'Nawala ang posisyon ko sa downsizing. Anong dokumento sa separation ang kailangan?', 'Nadula ang akon posisyon sa downsizing. Ano nga dokumento sa separation ang kinahanglan?'),
 ('Thanks for the introduction. How do I challenge an unjust dismissal?', 'Salamat sa pagpapakilala. Paano ko tututulan ang hindi makatarungang pagtanggal?', 'Salamat sa pagpakilala. Paano ko kontrahon ang indi makatarungan nga pagpahalin?')],
 '13th_month': [
 ('Do unpaid months reduce the proportional thirteenth month benefit?', 'Nababawasan ba ng mga buwang walang bayad ang proportional thirteenth month benefit?', 'Mabuhinan bala sang mga bulan nga wala bayad ang proportional thirteenth month benefit?'),
 ('I joined in September. How is my 13th month entitlement computed?', 'Setyembre ako nagsimula. Paano kukuwentahin ang 13th month entitlement ko?', 'Setyembre ako nagsugod. Paano kuwentahon ang akon 13th month entitlement?'),
 ('Which payslip components belong in the 13th month calculation?', 'Aling bahagi ng payslip ang kasama sa kuwenta ng 13th month?', 'Diin nga bahin sang payslip ang lakip sa pagkuwenta sang 13th month?')],
 'employment_status': [
 ('I sign a new short contract every harvest. Could I be a regular seasonal worker?', 'Pumipirma ako ng bagong maikling kontrata tuwing anihan. Maaari ba akong regular seasonal worker?', 'Nagapirma ako sang bag-o nga malip-ot nga kontrata kada ani. Mahimo bala ako regular seasonal worker?'),
 ('My work continues after the probation period. How is my status determined?', 'Tuloy ang trabaho ko pagkatapos ng probation period. Paano malalaman ang status ko?', 'Nagapadayon ang akon obra pagkatapos sang probation period. Paano mahibaluan ang akon status?'),
 ('Not a question about dismissal: what makes someone an independent contractor?', 'Hindi tungkol sa dismissal: ano ang batayan ng pagiging independent contractor?', 'Indi parte sa dismissal: ano ang basihan sang pagiging independent contractor?')],
 'occupational_safety': [
 ('A machine has exposed moving blades. Who should address this hazard?', 'May nakalantad na gumagalaw na talim ang makina. Sino ang dapat umaksyon sa panganib?', 'May nahayag nga nagahulag nga talim ang makina. Sin-o ang dapat mag-aksyon sa peligro?'),
 ('How do workers report a serious injury at the worksite?', 'Paano irereport ng manggagawa ang malubhang pinsala sa worksite?', 'Paano ireport sang mamumugon ang grabe nga pilas sa worksite?'),
 ('We were not taught how to handle hazardous waste. What training is required?', 'Hindi kami tinuruan sa paghawak ng hazardous waste. Anong training ang kailangan?', 'Wala kami gintudluan sa pag-uyat sang hazardous waste. Ano nga training ang kinahanglan?')],
 'holiday_pay': [
 ('I worked on a declared regular holiday. How should the payroll record that premium?', 'Pumasok ako sa idineklarang regular holiday. Paano itatala sa payroll ang premium?', 'Nagsulod ako sa gindeklarar nga regular holiday. Paano irekord sa payroll ang premium?'),
 ('How does an absence before a regular holiday affect holiday pay?', 'Paano naaapektuhan ng pagliban bago ang regular holiday ang holiday pay?', 'Paano makaapekto ang pagliban antes sang regular holiday sa holiday pay?'),
 ('Can you compare pay for a special non-working day and a regular holiday?', 'Maaari mo bang ihambing ang bayad sa special non-working day at regular holiday?', 'Pwede mo bala ikumparar ang bayad sa special non-working day kag regular holiday?')],
 'general_inquiry': [
 ('Good evening. How can employees raise a discrimination complaint?', 'Magandang gabi. Paano magrereklamo ang empleyado tungkol sa diskriminasyon?', 'Maayong gab-i. Paano magreklamo ang empleyado parte sa diskriminasyon?'),
 ('Please explain how an employee association is registered.', 'Pakipaliwanag kung paano magparehistro ng employee association.', 'Palihog ipaathag kon paano magparehistro sang employee association.'),
 ('Thanks, but I still need the rules on workplace privacy.', 'Salamat, pero kailangan ko pa ang patakaran sa privacy sa trabaho.', 'Salamat, pero kinahanglan ko pa ang patakaran sa privacy sa obra.')],
 'greeting': [
 ('A warm welcome to you, assistant.', 'Malugod na pagbati sa iyo, assistant.', 'Mainit nga pag-abiabi sa imo, assistant.'),
 ('Hi, pleased to meet you here!', 'Hi, masaya akong makilala ka rito!', 'Hi, nalipay ako nga makilala ka diri!'),
 ('Hello there, good to see you today.', 'Hello, nakakatuwang makita ka ngayon.', 'Hello, maayo nga makita ka subong.')],
 'small_talk': [
 ('Are you having a busy afternoon?', 'Marami ka bang ginagawa ngayong hapon?', 'Madamo ka bala ginahimo subong nga hapon?'),
 ('How are things going on your side today?', 'Kumusta ang mga bagay-bagay sa iyo ngayon?', 'Kamusta ang mga butang sa imo subong?'),
 ('I feel cheerful today. How are you feeling?', 'Masaya ako ngayon. Ano ang pakiramdam mo?', 'Nalipay ako subong. Ano ang imo pamatyag?')],
 'identity': [
 ('What should I call you in this chat?', 'Ano ang itatawag ko sa iyo sa chat na ito?', 'Ano ang itawag ko sa imo sa ini nga chat?'),
 ('Could you introduce the assistant behind these replies?', 'Maaari mo bang ipakilala ang assistant na nagbibigay ng mga sagot?', 'Pwede mo bala ipakilala ang assistant nga nagahatag sang mga sabat?'),
 ('Are you a chatbot answering automatically?', 'Chatbot ka bang awtomatikong sumasagot?', 'Chatbot ka bala nga awtomatiko nagasabat?')],
 'capability_question': [
 ('Which tasks can you assist users with?', 'Sa anong mga gawain ka makatutulong sa gumagamit?', 'Sa ano nga mga buluhaton ka makabulig sa nagagamit?'),
 ('Will you understand a question written in Hiligaynon?', 'Maiintindihan mo ba ang tanong na nakasulat sa Hiligaynon?', 'Mahangpan mo bala ang pamangkot nga nasulat sa Hiligaynon?'),
 ('What kinds of assistance fall outside your abilities?', 'Anong uri ng tulong ang lampas sa iyong kakayahan?', 'Ano nga klase sang bulig ang labaw sa imo masarangan?')],
 'gratitude': [
 ('I appreciate the time you took to clarify that.', 'Nagpapasalamat ako sa oras mo sa paglilinaw.', 'Nagapasalamat ako sa imo oras sa pagpaathag.'),
 ('That was just the explanation I needed. Thanks!', 'Iyon ang paliwanag na kailangan ko. Salamat!', 'Amo ina ang pagpaathag nga kinahanglan ko. Salamat!'),
 ('Many thanks for guiding me through this.', 'Maraming salamat sa paggabay sa akin dito.', 'Madamo nga salamat sa pagtuytoy sa akon diri.')],
 'farewell': [
 ('I need to leave this chat now. Take care!', 'Kailangan ko nang umalis sa chat. Ingat!', 'Kinahanglan ko na maghalin sa chat. Halong!'),
 ('We can continue tomorrow. Goodbye for now.', 'Ituloy natin bukas. Paalam muna.', 'Padayon kita buwas. Paalam anay.'),
 ('I am done for today, see you another day.', 'Tapos na ako ngayon, magkita tayo sa ibang araw.', 'Tapos na ako subong, kitaay kita sa iban nga adlaw.')],
}
