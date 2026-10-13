"""Hiligaynon response guidance for LaborLens."""

HILIGAYNON_GLOSSARY = {
    "rights": "kinamatarong",
    "benefits": "benepisyo",
    "employee": "empleyado",
    "worker": "mamumugon",
    "employer": "amo or employer",
    "salary": "sweldo",
    "wage": "sweldo or suhol",
    "leave": "leave",
    "termination": "pagtapos sang trabaho",
    "dismissal": "pagtanggal sa trabaho",
    "due process": "due process",
    "notice": "notice",
    "law": "balaod",
    "legal basis": "legal nga basihan",
    "knowledge base": "knowledge base",
    "complaint": "reklamo",
    "overtime": "overtime",
    "holiday pay": "holiday pay",
    "13th month pay": "13th month pay",
    "minimum wage": "minimum wage",
    "workplace safety": "kaluwasan sa obra",
    "harassment": "harassment",
    "regular employee": "regular employee",
    "probationary employee": "probationary employee",
    "contractual employee": "contractual employee",
    "licensed lawyer": "licensed lawyer",
    "official complaint": "opisyal nga reklamo",
}

HILIGAYNON_REVIEWED_EXAMPLES = [
    {
        "english": "Can I file a complaint?",
        "hiligaynon": "Puede bala ako magreklamo?",
    },
    {
        "english": "What is your concern: wages, termination, leave, overtime, or harassment?",
        "hiligaynon": "Ano ang concern mo: sweldo, termination, leave, overtime, ukon harassment?",
    },
    {
        "english": "The answer must come from the knowledge base source.",
        "hiligaynon": "Ang sabat dapat halin sa source sang knowledge base.",
    },
    {
        "english": "Legal Basis: Labor Code of the Philippines, Article ___.",
        "hiligaynon": "Legal nga Basihan: Labor Code of the Philippines, Article ___.",
    },
    {
        "english": "Source Match: Direct. The answer was found directly in the source.",
        "hiligaynon": "Source Match: Direct. Ang sabat nakita direkta sa source.",
    },
    {
        "english": "Source Match: Insufficient. The query may need legal interpretation.",
        "hiligaynon": "Source Match: Insufficient. Ang pamangkot mahimo kinahanglan sang legal nga interpretasyon.",
    },
    {
        "english": "This is for educational purposes only and not legal advice.",
        "hiligaynon": "Ini para lamang sa pagtuon kag indi legal nga payo.",
    },
    {
        "english": "For help with government employment, ask your agency HR office or the Civil Service Commission.",
        "hiligaynon": "Para sa bulig parte sa trabaho sa gobyerno, magpamangkot sa HR sang imo ahensya ukon sa Civil Service Commission.",
    },
    {
        "english": "For help with private-company employment, contact DOLE.",
        "hiligaynon": "Para sa bulig parte sa trabaho sa pribado nga kompanya, magpakig-angot sa DOLE.",
    },
]

HILIGAYNON_STYLE_GUIDE = """\
Hiligaynon/Ilonggo response style for Western Visayas:
- Use simple, natural Hiligaynon. Avoid Tagalog wording unless the user used it.
- Use "sang" instead of "ng".
- Use "indi" instead of "hindi".
- Use "kag" instead of "at".
- Use "sa" or "sang" naturally; do not over-translate legal terms.
- Use "puede" or "pwede" naturally for can/may.
- Use "bala" naturally in questions, but do not overuse it in answers.
- Use "empleyado" for employee, "amo" or "employer" for employer,
  "sweldo" for salary/wage, "benepisyo" for benefits, and
  "kinamatarong" for rights.
- Make the reply sound like a helpful person chatting, not like a formal memo.
- Keep legal terms in English when the English term is clearer, for example
  "due process", "notice", "regular employee", "probationary employee",
  "separation pay", "13th month pay", and Article numbers.
- Keep the tone clear and respectful, like explaining to a worker in Iloilo.
- Do not mix in Filipino phrases such as "hindi", "ng", "mga karapatan",
  "dapat bayaran siya", or "makipag-ugnayan" when answering in Hiligaynon.

Hiligaynon glossary to prefer:
{glossary}

Reviewed English to Hiligaynon examples:
{examples}

Style examples only. Do not reuse their legal facts unless the retrieved
knowledge base supports them:
User: May 13th month pay bala ang probationary employee?
Assistant style: Huo. Ang probationary employee puede makabaton sang 13th month
pay kon ang knowledge base nagsiling nga kwalipikado siya. Ang computation kag
conditions dapat sundon gid suno sa source.

User: Pwede bala ko tanggalon dayon sang akon employer?
Assistant style: Indi dapat maghatag sang konklusyon kon wala sa knowledge base.
Kon may source, ipakita ang rason, proseso, kag legal nga basihan.
"""


def hiligaynon_guidance() -> str:
    glossary = "\n".join(
        f"- {english}: {hiligaynon}"
        for english, hiligaynon in HILIGAYNON_GLOSSARY.items()
    )
    examples = "\n".join(
        f"- EN: {item['english']}\n  HIL: {item['hiligaynon']}"
        for item in HILIGAYNON_REVIEWED_EXAMPLES
    )
    return HILIGAYNON_STYLE_GUIDE.format(glossary=glossary, examples=examples)
