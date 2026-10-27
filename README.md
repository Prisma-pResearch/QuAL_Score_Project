# QuAL Score Project
This repository describes the development of two models for automatically assigning Quality of Assessment for Learning (QuAL) scores to de-identified surgical education feedback. 

## Overview
**Surgical Education Feedback** consists of written or voice-dictated feedback from surgical faculty to trainees. Narrative feedback quality in surgical Entrustable Professional Activity (EPA) assessments is variable, limiting consistent evaluation and opportunities for improvement. We evaluated whether fine-tuned artificial intelligence models could reproduce expert-assigned feedback quality scores and whether a simplified scoring rubric improved assessment consistency.

Models tested: 
  1) Pre-trained clinical transformer: BioClinicalBERT
  2) LLM-as-a-Judge (LLMaaJ) method

Data: 1509 feedback comments that were rated by two domain experts. Each comment was scored using the original QuAL score, a 0-5 point scale rating system based on the summation of a score of evidence (0-3 points), suggestion (0-1 point), and connection (0-1 point). If the total QuAL scores showed a difference of more than 1 point between the two raters, that comment was then scored by a third rater. We also piloted a modified rubric (mQuAL) that assigned 1 point per domain on a 3-point scale.

## Results 
**Overall interrater reliability was high (κ = 0.85), although raters disagreed on 27.8% of QuAL evidence scores. 

**Binary evidence scoring with mQuAL reduced disagreement to 8.5%. 

**Encoder and generative models achieved similar accuracy for suggestion (0.96 versus 0.94) and connection (0.95 vs 0.91), whereas the encoder outperformed the generative model for evidence (0.93 vs 0.77). 

**Human rater agreement and model performance were both strongest with the mQuAL rubric that assigns 1 point per domain on a 30-oint scale

## Repository Structure
```
philter_surgical_education_feedback/
├── main.py                                       # Core philter entry point
├── pyproject.toml                                # Package configuration (v0.2.1)
├── requirements.txt                              # Python dependencies
├── LICENSE                                       # License file
├── README.md                                     # This file
│
├── src/philter/                                  # Installable Python package
│   ├── __init__.py                               # Re-exports Philter, CoordinateMap
│   ├── core.py                                   # Philter engine (was philter.py)
│   ├── coordinate_map.py                         # Coordinate utilities
│   └── data/
│       ├── configs/
│       │   ├── philter_delta.json                # Original philter config (stage 1)
│       │   └── philter_customized_feedback.json  # Custom config for surgical education feedback (stage 3)
│       └── filters/
│           ├── regex/
│           │   └── safe/
│           │       └── customized_feedback_words.txt  # Allowed words whitelist
│           ├── blacklists/                       # PHI name dictionaries
│           └── whitelists/                       # Protected terms
│
├── workflows/
│   ├── deidentify_feedback.py                    # De-identification workflow (Stage 1 and Stage 3)
│   └── analyze_overdeid_feedback.py              # Over-deidentification analysis (Stage 2)
│
├── benchmarks/                                   # Performance benchmarks (pytest-benchmark)
│   ├── test_file_throughput.py
│   ├── test_inmemory_feasibility.py
│   └── test_memory.py
│
├── docs/
│   └── architecture.md                           # Architecture documentation
│
├── validation/
│   ├── gpt_oss_phi_review_prompt.txt             # The exact prompt used       
│   └── gpt_oss_phi_review_5000_results.csv       # Full results with all 5000 rows         
│
└── examples/
    └── pipeline_comparison_examples.csv          # 100 examples comparing philter-ucsf vs philter_surgical_education_feedback

```

## Usage
### Prerequisites
- Python 3.11 or higher
- pip, conda, or [uv](https://docs.astral.sh/uv/) package manager
  
### Command Line Installation
```bash
# Clone the repository
git clone https://github.com/Prisma-pResearch/QuAL_Score_Project.git
cd QuAL_Score_Project
```
```bash
# Install dependencies (pick one)
pip install -r requirements.txt    # pip
uv sync                            # uv (uses pyproject.toml + uv.lock)
```

### Running Tests and Benchmarks
```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/ -v

# Run benchmarks
uv run pytest benchmarks/ --benchmark-enable -v

# Compare against saved baseline
uv run pytest benchmarks/ --benchmark-enable --benchmark-compare=0001
```

## Quick Start
Once installed, proceed to the workflow steps below.

### Workflow: Surgical Education Feedback (CSV Data)
####  Stage 1: Initial De-identification
The script accepts a single CSV file or a directory of multiple CSV files as input. Courtesy of brianceorge at [SIMPL](https://www.simpl.org/), the loading method clears internal state while preserving compiled patterns to allow efficient loading of up to 1M rows of feedback comments.
```bash
# Single CSV file
python workflows/deidentify_feedback.py --input "path/to/feedback.csv" --output "output_stage1" --config initial


**Example Output**
At this stage, PHI has removed and some terms that are not PHI have also removed and must be restored by subsequent steps:
```
| value_deidentified
|------------------------------------------------------------
| ****** is an exceptional teacher- constantly
| yearning for knowledge and sharing it with peers.
|------------------------------------------------------------
| Good job with dissection and ******. Robotic
| Skills improved substantially over the month.
|------------------------------------------------------------
| Dr. *** is an eager and enthusiastic ******* and
| adapts quickly to our techniques in the OR.
```

#### Stage 2: Analyze Over-deidentification (Optional)

**Note:** Stage 2 is optional. To identify words incorrectly flagged as PHI,  
- If you want to use our [pre-built allowed words list](https://github.com/Prisma-pResearch/philter_surgical_education_feedback/blob/main/src/philter/data/filters/regex/safe/customized_feedback_words.txt) that was extended through a pull request by briancgeorge at [SIMPL](https://www.simpl.org/) with additional surgical instrument names and medical terminology common in operative assessments (Babcock, Metzenbaum, Harmonic, etc.), skip directly to **Stage 3 Step 2: Re-run Pipeline** to run with our custom configuration `--config updated`.

- If you want to customize the allowed words list for your own dataset, run:

```bash
# Default: top 60 words
python workflows/analyze_overdeid_feedback.py "output_stage1/*_deidentified.csv"

# Custom: specify your own n
python workflows/analyze_overdeid_feedback.py "output_stage1/*_deidentified.csv" --top_n 30
```
**Example Output**

#### Stage 3: Refine De-identification
**Step 1: Update Configuration**

Add the allowed words pattern at the beginning of the filters array in `src/philter/data/configs/philter_customized_feedback.json`
  ``` json
   {
        "title": "Customized Words Safe",
        "type": "regex",
        "exclude": false,
        "filepath": "src/philter/data/filters/regex/safe/customized_feedback_words.txt",
        "notes": "Common surgical education terms that should not be redacted"
    },
 ```
**Step 2: Re-run Pipeline**
 ``` bash
python workflows/deidentify_feedback.py --input "path/to/csvs/" --output "output_stage3" --config updated
 ```

**Example Output:**
At this stage, PHI has removed and some terms that were incorrectly redacted ("sewing" and "learner") have been restored:
```
| value_deidentified
|------------------------------------------------------------
| ****** is an exceptional teacher- constantly
| yearning for knowledge and sharing it with peers.
|------------------------------------------------------------
| Good job with dissection and sewing. Robotic
| Skills improved substantially over the month.
|------------------------------------------------------------
| Dr. *** is an eager and enthusiastic learner and
| adapts quickly to our techniques in the OR.
```


### Summary
Fine-tuned models reproduced expert assessments of surgical narrative feedback with high domain-specific accuracy, particularly using encoder models. Simplifying evidence scoring to the 3-point scale improved interrater agreement and model performance, supporting the use of AI-assisted feedback quality assessment within surgical residency programs to identify low-quality feedback and initiate targeted interventions.


## Limitations
- **Incomplete PHI coverage**: The pipeline inherits philter-ucsf's  blacklist-based limitations - uncommon names (e.g., 'Narica') and institution names (e.g., "Scottish Rite") may not be flagged if 
  absent from philter's blacklists.
- **Partial over-deidentification correction**: Stage 2 identifies only the most frequently over-redacted words, so less frequent surgical education terms may still be over-redacted.
- **LLM validation false positives**: LLMs may flag non-PHI content as potential PHI regardless of prompt specificity. Manual review of flagged samples is recommended as a complementary step.


## Maintainers and Contributors
- Die Hu (hudier1007)
- Sasank Desaraju (sasank-desaraju)
- Victoria Nolan (victorianolan8)
- Tyler Loftus (tjloftus)


## Support
For questions about this project: contact the repository maintainers


## License
See the [LICENSE](https://github.com/Prisma-pResearch/QuAL_Score_Project/blob/main/LICENSE) for details.


## Disclaimer
This software is provided "as is", without warranty of any kind. The authors of this work will not be liable for any claim, damages or other liability arising in connection with this software.
