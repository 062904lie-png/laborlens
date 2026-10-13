"""Convert supported model math markup to readable plain-text formulas."""
import re


def normalize_formula_markup(answer: str) -> str:
    def math_text(match):
        text = match.group(1)
        text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\(?:bigl|bigr|Bigl|Bigr|left|right)\b", "", text)
        text = re.sub(r"\\(?:max|min)\b", lambda m: m.group()[1:], text)
        text = re.sub(r"\\(?:times|cdot)\b", "×", text)
        text = re.sub(r"\\div\b", "÷", text)
        text = re.sub(r"\\(?:frac|dfrac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\[,;:!]|\\(?:quad|qquad)\b", " ", text)
        text = re.sub(r"\\[ \t]*(?=\n|$)", "", text)
        return re.sub(r"\s+", " ", text).strip()

    # Preserve code examples/paths; modify only explicit math spans otherwise.
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]*`)", answer or "")
    for index in range(0, len(parts), 2):
        text = parts[index]
        for pattern in (r"\\\[([\s\S]*?)\\\]", r"\\\(([\s\S]*?)\\\)", r"\$\$([\s\S]*?)\$\$"):
            text = re.sub(pattern, math_text, text)
        # Markdown hard-break backslashes are presentation syntax, not content.
        text = re.sub(r"\\[ \t]*(?=\r?\n|$)", "", text)
        parts[index] = text
    return "".join(parts)
