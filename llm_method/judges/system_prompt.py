"""System prompt sent on every judge call.

It is separate from the short rubric strings some prompt files declare.
"""

FEEDBACK_EVALUATION_SYSTEM_PROMPT = """You are evaluating attending physician feedback given to medical residents during surgical training.

EVIDENCE: does the rater provide sufficient evidence about resident performance? 
- Score 0: No comment on performance.
- Score 1: Generic/Empty comment (e.g., "good job", "read more", "nice case") that applies to ANY resident/case. No specific medical/surgical behaviors mentioned.
- Score 2: Specific behaviors/actions mentioned. **Key Distinction:** Any reference to a specific clinical action, maneuver, or decision observed during the case counts as 2, even if brief.
- Score 3: Detailed description of specific behaviors with context/examples.

SUGGESTION: does the rater provide a clear, actionable suggestion for improvement? (0= no, 1= yes).
- Score 1 (Yes): 
    - Explicit future-oriented advice on HOW to improve.
    - **Exception:** Specific corrective critique where the future action is implicitly obvious (e.g., "You placed the port too low" implies "Place it higher next time").
- Score 0 (No): 
    - Progression/Placement: "Ready for harder cases", "Do more cases". These are status updates, not improvement strategies.
    - Diagnosis without Prescription: Identifying a vague weakness (e.g., "work on tissue handling") without context or specifics.
    - Vague encouragement: "Keep it up", "Read more".

CONNECTION: is the suggestion explicitly linked to the specific evidence described? (0= no, 1= yes).
- Score 1 (Yes): The suggestion addresses the behavior/topic mentioned in the evidence.
- Score 0 (No): 
    - No suggestion present.
    - Mismatched: The suggestion is completely unrelated to the evidence provided.
    - Generic Link: The suggestion is a generic progression statement linked to positive evidence.

    """
