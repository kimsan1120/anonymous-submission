# Paper Prompt Export

This directory contains Korean prompt originals separated for paper release.

## 01_pii_redaction

- `pii_redaction_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/pii_remove.txt`
  - Purpose: PII/sensitive-information de-identification for SMS and ASR call transcripts.

## 02_evidence_annotation

- `sms_evidence_annotation_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/sms_teacher_functional_label01_extract.txt`
  - Purpose: functional evidence span annotation for smishing messages.
- `voice_evidence_annotation_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/voice_teacher_functional_label01_extract.txt`
  - Purpose: functional evidence span annotation for vishing transcripts.

## 03_rationale_generation

- `sms_rationale_system_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_sms_sys.txt`
- `sms_rationale_user_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_pseudo_sms_user.txt`
- `voice_rationale_system_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_voice_sys.txt`
- `voice_rationale_user_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_pseudo_voice_user.txt`

## 04_inference_templates

- `sms_binary_inference_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/sms_binary_short.txt`
- `voice_binary_inference_prompt_ko.txt`
  - Source: `src/phishdec/prompts/instructions/voice_binary_short.txt`
- `sms_label_explanation_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_first_sms_sys.txt`
- `voice_label_explanation_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_first_voice_sys.txt`
- `sms_label_evidence_only_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_only_sms_sys.txt`
- `voice_label_evidence_only_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_only_voice_sys.txt`
- `sms_label_evidence_explanation_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_sms_sys.txt`
- `voice_label_evidence_explanation_template_ko.txt`
  - Source: `src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_voice_sys.txt`
