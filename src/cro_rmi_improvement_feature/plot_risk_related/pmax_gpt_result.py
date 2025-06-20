# risk_interdependency_llm.py

import openai
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

# Load your OpenAI API key securely
openai.api_key = "sk-..."  # Replace with your API key or use environment variable

# === Data ===

risk_descriptions = {
    "R001": "Phishing attacks targeting employee emails",
    "R002": "Credential reuse due to poor password hygiene",
    "R003": "Insider misuse of admin access to exfiltrate sensitive data",
    "R004": "Delayed revocation of credentials after employee termination",
    "R005": "Privilege escalation vulnerability in legacy systems",
    "R006": "Weak audit trail for access to critical systems",
}

risk_pairs = [
    ("R002", "R001"),
    ("R004", "R003"),
    ("R003", "R006"),
    ("R005", "R004"),
]

# === Pydantic Model ===


class RiskRelationResult(BaseModel):
    pair_id: str
    interdependency_type: str = Field(
        ...,
        regex=r"^(Causal|Contingent|Shared Root Cause|Correlated|Temporal/Sequential|None)$",
    )
    direction: Optional[str] = Field(None, regex=r"^(A → B|B → A|None)$")
    rationale: str
    confidence: int = Field(..., ge=1, le=5)


# === Prompt Generation ===


def generate_prompt(pair_id, risk_a, risk_b, desc_a, desc_b):
    return f"""
You are a risk analysis assistant reviewing pairs of risk statements.
Analyze the interdependency between the following two risks:

[Pair ID: {pair_id}]
Risk A: {desc_a}
Risk B: {desc_b}

Determine:
1. Interdependency Type
2. Direction (if directional)
3. Rationale (1-2 sentences)
4. Confidence (1-5)

Valid Types:
- Causal
- Contingent
- Shared Root Cause
- Correlated
- Temporal/Sequential
- None

If type is Causal, Contingent, or Temporal/Sequential, include direction:
A → B or B → A. Otherwise, use "None".

Respond in JSON:
{{
  "pair_id": "{pair_id}",
  "interdependency_type": "...",
  "direction": "...",
  "rationale": "...",
  "confidence": ...
}}
"""


# === Call OpenAI and Parse ===


def analyze_pair(pair_id, risk_a, risk_b, desc_a, desc_b):
    prompt = generate_prompt(pair_id, risk_a, risk_b, desc_a, desc_b)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = response.choices[0].message["content"].strip()

    try:
        result = RiskRelationResult.model_validate_json(content)
        return result
    except ValidationError as e:
        print(f"[ERROR] Failed to parse or validate LLM output for {pair_id}: {e}")
        print("Raw output:", content)
        return None


# === Main Loop ===

if __name__ == "__main__":
    results = []

    for ra, rb in risk_pairs:
        pair_id = f"{ra} & {rb}"
        desc_a = risk_descriptions.get(ra, "Unknown")
        desc_b = risk_descriptions.get(rb, "Unknown")

        print(f"\n🔎 Analyzing {pair_id}...")
        result = analyze_pair(pair_id, ra, rb, desc_a, desc_b)

        if result:
            results.append(result)
            print(result.model_dump_json(indent=2))

    # TODO: optionally export results to JSON, CSV, or Excel
