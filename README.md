# QuAL Score Project
This repository describes the development of two models for automatically assigning Quality of Assessment for Learning (QuAL) scores to de-identified surgical education feedback. 

## Overview
**Surgical Education Feedback** consists of written or voice-dictated feedback from surgical faculty to trainees. Narrative feedback quality in surgical Entrustable Professional Activity (EPA) assessments is variable, limiting consistent evaluation and opportunities for improvement. We evaluated whether fine-tuned artificial intelligence models could reproduce expert-assigned feedback quality scores and whether a simplified scoring rubric improved assessment consistency.

Models tested: 
  1) Pre-trained clinical transformer: BioClinicalBERT
  2) LLM-as-a-Judge (LLMaaJ) method

Data: 1509 feedback comments that were rated by two domain experts. Each comment was scored using the original QuAL score, a 0-5 point scale rating system based on the summation of a score of evidence (0-3 points), suggestion (0-1 point), and connection (0-1 point). If the total QuAL scores showed a difference of more than 1 point between the two raters, that comment was then scored by a third rater. We also piloted a modified rubric (mQuAL) that assigned 1 point per domain on a 3-point scale.

## Results 
- Overall interrater reliability was high (κ = 0.85), although raters disagreed on 27.8% of QuAL evidence scores. 

- Binary evidence scoring with mQuAL reduced disagreement to 8.5%. 

- Encoder and generative models achieved similar accuracy for suggestion (0.96 versus 0.94) and connection (0.95 vs 0.91), whereas the encoder outperformed the generative model for evidence (0.93 vs 0.77). 

- Human rater agreement and model performance were both strongest with the mQuAL rubric that assigns 1 point per domain on a 3 point scale

## Repository Structure
```text
QuAL_Score_Project/
├── BIOCLINICALBERT_Ordinal.ipynb
├── BIOCLINICALBERT_Binary.ipynb
├── QUAL_Inference_Only.ipynb
├── checkpoints/
│   ├── best_model_ordinal.pt
│   └── best_model_binary.pt
├── llm_method/
├── BIOCLINICALBERT_Methods&Results.docx
├── README.md
└── LICENSE
```
- `BIOCLINICALBERT_Ordinal.ipynb` trains the ordinal encoder used to generate the 5-point QuAL score.
- `BIOCLINICALBERT_Binary.ipynb` trains the binary encoder used to generate the 3-point mQuAL score.
- `QUAL_Inference_Only.ipynb` applies the locked model checkpoints to new Excel files without retraining.
- `checkpoints/` contains the locked ordinal and binary model weights.
- `llm_method/` contains the LLM-as-a-Judge implementation.

## BioClinicalBERT Encoder
The encoder approach uses `emilyalsentzer/Bio_ClinicalBERT`, a transformer model pretrained on clinical text. Each feedback comment is tokenized and passed to a multi-task model with separate output heads for evidence, suggestion, and connection.

### Ordinal Model
`BIOCLINICALBERT_Ordinal.ipynb` predicts:
- Evidence: 0–3
- Suggestion: 0–1
- Connection: 0–1
- QuAL total: 0–5
  
The total is returned as `five_point_qual_score`.
### Binary Model
`BIOCLINICALBERT_Binary.ipynb` predicts:
- Evidence: 0–1
- Suggestion: 0–1
- Connection: 0–1
- mQuAL total: 0–3
  
For binary Evidence, original Evidence scores of 0–1 are mapped to 0, while scores of 2–3 are mapped to 1. The three binary predictions are summed and returned as `three_point_qual_score`.

Both models use an  80/10/10 train, validation, and test split.  Training uses AdamW optimization for up to 50 epochs, with early stopping after 10 epochs without improvement. The locked checkpoints correspond to the models with the lowest validation total-score MSE.

## Inference
Use `QUAL_Inference_Only.ipynb` to score new Excel workbooks with the locked binary and ordinal checkpoints.
### Input
- An `.xlsx` workbook containing a column named `Feedback`
- The target worksheet and file path are configured in the notebook
- Feedback must be de-identified before processing
  
Blank, symbol-only, and single-character entries are not scored and remain blank in the output.
### Output
The notebook creates a new file ending in `_QUAL_scored.xlsx` and adds:
| Output column | Model | Score |
|---|---|---|
| `three_point_qual_score` | Binary encoder | mQuAL total, 0–3 |
| `five_point_qual_score` | Ordinal encoder | QuAL total, 0–5 |

The source workbook is not overwritten. Existing worksheets and workbook formatting are preserved.

### Usage
1. Set the input workbook path and target worksheet in `QUAL_Inference_Only.ipynb`.
2. Confirm that the binary and ordinal checkpoint paths are correct.
3. Run the inference notebook.
4. Review the generated `_QUAL_scored.xlsx` file.
   
Use the locked checkpoints for future inference runs. Retraining is not required when scoring new feedback files.

## Summary
Fine-tuned models reproduced expert assessments of surgical narrative feedback with high domain-specific accuracy, particularly using encoder models. Simplifying evidence scoring to the 3-point scale improved interrater agreement and model performance, supporting the use of AI-assisted feedback quality assessment within surgical residency programs to identify low-quality feedback and initiate targeted interventions.

## Maintainers and Contributors
- Die Hu (hudier1007)
- Michael Kabbash (kabbashm)
- Sasank Desaraju (sasank-desaraju)
- Victoria Nolan (victorianolan8)
- Tyler Loftus (tjloftus)

## Support
For questions about this project: contact the repository maintainers

## License
See the [LICENSE](https://github.com/Prisma-pResearch/QuAL_Score_Project/blob/main/LICENSE) for details.

## Disclaimer
This software is provided "as is", without warranty of any kind. The authors of this work will not be liable for any claim, damages or other liability arising in connection with this software.
