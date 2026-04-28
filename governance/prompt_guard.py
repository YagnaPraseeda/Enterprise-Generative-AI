"""
Prompt Injection Guard
Detects and blocks prompt injection attacks, jailbreak attempts,
and other adversarial prompt patterns.
"""

import logging
import re

logger = logging.getLogger(__name__)


class PromptGuard:
    """
    Protects against prompt injection attacks using pattern matching
    and heuristic analysis.

    Detects:
    - System prompt extraction attempts
    - Instruction override attacks
    - Role impersonation
    - Data exfiltration attempts
    - Jailbreak patterns
    """

    # Dangerous patterns (case-insensitive regex)
    INJECTION_PATTERNS = [
        # Instruction override
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
         "instruction_override"),
        (r"disregard\s+(all\s+)?(previous|prior|above)",
         "instruction_override"),
        (r"forget\s+(everything|all|your)\s+(previous|prior|instructions?)",
         "instruction_override"),

        # System prompt extraction
        (r"(show|reveal|display|print|output|repeat)\s+(me\s+)?(the\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
         "system_prompt_extraction"),
        (r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
         "system_prompt_extraction"),

        # Role impersonation
        (r"you\s+are\s+now\s+(a|an|the)\s+",
         "role_impersonation"),
        (r"pretend\s+(to\s+be|you\s+are)\s+",
         "role_impersonation"),
        (r"act\s+as\s+(a|an|the)\s+(?!enterprise|assistant|helpful)",
         "role_impersonation"),

        # Data exfiltration
        (r"(give|show|provide|send)\s+(me\s+)?(all\s+)?(confidential|secret|private|internal)\s+(data|info|information)",
         "data_exfiltration"),
        (r"(extract|dump|export)\s+(all\s+)?(the\s+)?(data|information|records)",
         "data_exfiltration"),

        # Jailbreak patterns
        (r"DAN\s+mode",
         "jailbreak"),
        (r"developer\s+mode\s+(enabled|on|activated)",
         "jailbreak"),
        (r"(enable|activate|enter)\s+(unrestricted|unfiltered|unlimited)\s+mode",
         "jailbreak"),

        # Encoding attacks
        (r"(base64|rot13|hex)\s*(encode|decode|convert)",
         "encoding_attack"),
    ]

    # Suspicious keywords that raise alert level
    SUSPICIOUS_KEYWORDS = [
        "sudo", "admin", "root", "password", "credential",
        "api_key", "api key", "secret_key", "token", "hack",
        "bypass", "override", "injection",
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), risk_type)
            for pattern, risk_type in self.INJECTION_PATTERNS
        ]

    def check(self, prompt: str) -> dict:
        """
        Check a user prompt for injection attacks.

        Returns:
            dict with keys:
                - is_safe (bool): whether the prompt is safe to process
                - risk_type (str): type of risk detected (if any)
                - risk_score (float): 0.0-1.0 risk score
                - explanation (str): human-readable explanation
                - flagged_patterns (list): list of matched patterns
        """
        flagged_patterns = []
        risk_types = set()

        # Check regex patterns
        for compiled, risk_type in self._compiled_patterns:
            if compiled.search(prompt):
                flagged_patterns.append(risk_type)
                risk_types.add(risk_type)

        # Check suspicious keywords
        prompt_lower = prompt.lower()
        suspicious_found = [
            kw for kw in self.SUSPICIOUS_KEYWORDS
            if kw in prompt_lower
        ]

        # Calculate risk score
        risk_score = 0.0
        if flagged_patterns:
            risk_score = min(1.0, len(flagged_patterns) * 0.4)
        if suspicious_found:
            risk_score = min(1.0, risk_score + len(suspicious_found) * 0.1)

        is_safe = len(flagged_patterns) == 0
        primary_risk = list(risk_types)[0] if risk_types else "none"

        if not is_safe:
            explanation = (
                f"🛡️ Prompt injection detected! "
                f"Type: {primary_risk}. "
                f"Flagged patterns: {', '.join(set(flagged_patterns))}. "
                f"Risk score: {risk_score:.0%}"
            )
        elif suspicious_found:
            explanation = (
                f"⚠️ Prompt contains suspicious keywords: "
                f"{', '.join(suspicious_found)}. "
                f"Proceeding with caution."
            )
        else:
            explanation = "✅ Prompt appears safe."

        return {
            "is_safe": is_safe,
            "risk_type": primary_risk,
            "risk_score": round(risk_score, 2),
            "explanation": explanation,
            "flagged_patterns": list(set(flagged_patterns)),
            "suspicious_keywords": suspicious_found,
        }
