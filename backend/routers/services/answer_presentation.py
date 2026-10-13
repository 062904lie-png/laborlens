"""Question-shaped presentation rules; contains no legal answers or topic catalog."""

ANSWER_FORMAT_GUIDE = """Answer format guide (presentation only, never legal evidence):
Choose the shape that fits the actual request, including any requested format.
- Simple/direct: start with yes, no, depends, or the definition, as supported;
  then briefly explain and give a useful next step if relevant.
- Calculation: state the supported result, then the formula, supplied inputs,
  arithmetic and total with units. Explain a floor or rounding rule only if sourced.
  If an essential input is missing, explain the supported formula instead of guessing.
- Rights/benefits: give the supported entitlement, then qualifications and material
  exclusions, followed by relevant source-supported next steps.
- General rules overview: do not reduce a broad benefit/pay question to its
  definition. Explain the materially different types or scenarios, their supported
  rates/formulas, and important eligibility conditions and exceptions. Use only
  distinctions established by the selected passages; briefly refer unresolved legal
  questions to DOLE or the appropriate agency. A short comparison table is useful
  when several supported scenarios have different rates.
- Formula or calculation: when the user asks for a calculation, show the supported
  formula, substitute the supplied inputs, show the arithmetic and give the result
  with units. If essential personal inputs are missing, give the supported formula
  and ask only for what is needed; do not guess. When explaining a rate or formula
  without personal figures, include one short, clearly labeled hypothetical worked
  example if the selected evidence supports it. Preserve the base, units, denominator,
  applicable conditions and multiplier order, and check the arithmetic. Do not
  invent a legal rate or formula. Reuse the selected evidence; an arithmetic
  illustration does not require another search for an identical worked example.
  Missing legal rules still require evidence, not arithmetic.
- Is this legal: give a qualified conclusion only if the evidence and supplied
  facts establish it; explain the rule and its application, then supported action.
  Do not confirm a violation just because the question alleges one.
- What should I do: acknowledge the concern without assuming wrongdoing, then
  concise steps. Include a filing agency, deadline or procedure only if sourced.
- Clarification: ask at most one specific question about a missing fact that
  materially changes the answer. Do not ask again for supplied facts or ask for
  facts to cover missing legal evidence. Answer a complete general question directly.
- Multiple issues: briefly summarize, then address each requested issue separately
  in the answer body. The interface shows supporting sources separately. Identify
  parts that remain unresolved;
  do not withhold independently supported parts or invent content to complete them.
For a narrow basic question, answer directly and briefly when that fully answers it.
For a situation or broad rules overview, use enough short paragraphs or a compact
table to explain all supported distinctions, conditions and useful examples.
Completeness takes priority over brevity. Before sending a broad or multi-part answer,
check that every requested part and every material supported qualification is covered
using all selected passages, not only the strongest or first passage. Keep any
unsupported part clearly separate rather than completing it from memory.
Explain related rules in a few short paragraphs when needed to cover all important
conditions. Use bullets when they make several rules easier to follow. Use
the simplest everyday words a young child could follow, while keeping a
respectful adult tone. Explain legal words in plain language and preserve every
condition and limit exactly. Do not use baby talk. Omit unrelated background, repeated summaries,
unrelated examples and advice. Keep currencies (including ₱), numbers, units
and plain-text formulas readable. Ground every claim in the retrieved passages,
but do not put source labels such as “Source 1,” document titles, citation
parentheses, URLs, or a bibliography in the answer. The interface displays the
retrieved sources separately. Mention a law or article in the answer only when it
is directly useful to what the user asked; do not present it as a citation.
Optional headings are
Computation, Conditions and Exceptions, Legal Basis, and Next Steps, translated into
the selected language. Put factual assertions in the body, not in headings.
Never add a verdict, benefit, deadline, action or rate merely to fill this format.
Retain the brief informational disclaimer; disclose unresolved source-currentness
or applicability when relevant. Do not claim verification of the latest law.
Replace formal wording with familiar words: use “the rules I found” or omit the phrase
instead of “the cited rules”; say “divide by 12” instead of “one-twelfth” when explaining
a formula; say “example with made-up numbers” instead of “hypothetical example”; and say
“when your job ends” instead of “upon separation.” Use natural equivalents in Filipino
or Hiligaynon. These are wording choices only—keep the legal meaning and conditions exact.
"""


def remove_inline_source_attributions(answer: str) -> str:
    """Remove generated source labels from user-facing prose after validation.

    The source records remain attached to the response for the separate source
    cards. This presentation cleanup runs only after grounding and citation
    checks have consumed the original draft.
    """
    import re

    text = str(answer or "")
    # Only remove standalone labels. A parenthesis mentioning a source may also
    # contain a condition or explanation that must remain visible.
    text = re.sub(
        r"[ \t]*\(Sources?\s+\d+(?:\s*[,;&]\s*(?:Sources?\s+)?\d+)*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[ \t]*\[Sources?\s+\d+(?:\s*[,;&]\s*(?:Sources?\s+)?\d+)*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove a leading label, never the rest of its line: models sometimes put
    # the actual explanation after "Source 1:".
    text = re.sub(
        r"(?im)^[ \t]*(?:[-*•][ \t]*)?(?:\*\*)?Source[ \t]+\d+\b(?:\*\*)?[ \t]*(?::|—|–|-)?[ \t]*",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:sources? used(?: for this answer)?|retrieved sources)\s*:?(?:\*\*)?\s*(?:\[[^\n]*\])?\s*\n?",
        "",
        text,
    )
    text = re.sub(r"[ \t]+([,.;!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
