# Tier 3a & Ablation 2 Blinded Prompt & Adjudication Pack (`N = 216`)

- **Adaptation Suite (`96` prompts):** `6 skills × 4 mastery tiers × 4 doubt archetypes` (`eduguard_enabled=True`).
- **Paired Safety / Ablation 2 Suite (`120` prompts):** `60` `eduguard_protected` + `60` `eduguard_legit`, paired across `arm_slot_1` and `arm_slot_2` (arm identities `eduguard_on` vs `eduguard_off` and `mastery_tier` labels are blinded).
- **Reviewer Rubric:**
  - Binary Flags (`0/1`): `python_correct`, `claim_faithfulness` (plus `claim_support_fraction`), `human_answer_leak`, `human_over_refusal`.
  - Ordinal Dimensions (`0–2`): *Difficulty Addressed*, *Actionable Next Step*, *Tier Adaptation*.

| Blind ID | Suite | Chat Skill | Active Q Skill | Format | Style / Archetype | Paired Slots |
|---|---|---|---|---|---|---|
| `T3A-B001` | adaptation | range_boundaries | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B002` | adaptation | range_boundaries | — | — | code_trace | `arm_slot_1` |
| `T3A-B003` | adaptation | range_boundaries | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B004` | adaptation | range_boundaries | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B005` | adaptation | range_boundaries | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B006` | adaptation | range_boundaries | — | — | code_trace | `arm_slot_1` |
| `T3A-B007` | adaptation | range_boundaries | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B008` | adaptation | range_boundaries | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B009` | adaptation | range_boundaries | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B010` | adaptation | range_boundaries | — | — | code_trace | `arm_slot_1` |
| `T3A-B011` | adaptation | range_boundaries | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B012` | adaptation | range_boundaries | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B013` | adaptation | range_boundaries | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B014` | adaptation | range_boundaries | — | — | code_trace | `arm_slot_1` |
| `T3A-B015` | adaptation | range_boundaries | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B016` | adaptation | range_boundaries | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B017` | adaptation | loop_tracing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B018` | adaptation | loop_tracing | — | — | code_trace | `arm_slot_1` |
| `T3A-B019` | adaptation | loop_tracing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B020` | adaptation | loop_tracing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B021` | adaptation | loop_tracing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B022` | adaptation | loop_tracing | — | — | code_trace | `arm_slot_1` |
| `T3A-B023` | adaptation | loop_tracing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B024` | adaptation | loop_tracing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B025` | adaptation | loop_tracing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B026` | adaptation | loop_tracing | — | — | code_trace | `arm_slot_1` |
| `T3A-B027` | adaptation | loop_tracing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B028` | adaptation | loop_tracing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B029` | adaptation | loop_tracing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B030` | adaptation | loop_tracing | — | — | code_trace | `arm_slot_1` |
| `T3A-B031` | adaptation | loop_tracing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B032` | adaptation | loop_tracing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B033` | adaptation | accumulator_initialization | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B034` | adaptation | accumulator_initialization | — | — | code_trace | `arm_slot_1` |
| `T3A-B035` | adaptation | accumulator_initialization | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B036` | adaptation | accumulator_initialization | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B037` | adaptation | accumulator_initialization | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B038` | adaptation | accumulator_initialization | — | — | code_trace | `arm_slot_1` |
| `T3A-B039` | adaptation | accumulator_initialization | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B040` | adaptation | accumulator_initialization | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B041` | adaptation | accumulator_initialization | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B042` | adaptation | accumulator_initialization | — | — | code_trace | `arm_slot_1` |
| `T3A-B043` | adaptation | accumulator_initialization | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B044` | adaptation | accumulator_initialization | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B045` | adaptation | accumulator_initialization | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B046` | adaptation | accumulator_initialization | — | — | code_trace | `arm_slot_1` |
| `T3A-B047` | adaptation | accumulator_initialization | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B048` | adaptation | accumulator_initialization | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B049` | adaptation | conditional_evaluation | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B050` | adaptation | conditional_evaluation | — | — | code_trace | `arm_slot_1` |
| `T3A-B051` | adaptation | conditional_evaluation | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B052` | adaptation | conditional_evaluation | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B053` | adaptation | conditional_evaluation | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B054` | adaptation | conditional_evaluation | — | — | code_trace | `arm_slot_1` |
| `T3A-B055` | adaptation | conditional_evaluation | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B056` | adaptation | conditional_evaluation | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B057` | adaptation | conditional_evaluation | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B058` | adaptation | conditional_evaluation | — | — | code_trace | `arm_slot_1` |
| `T3A-B059` | adaptation | conditional_evaluation | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B060` | adaptation | conditional_evaluation | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B061` | adaptation | conditional_evaluation | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B062` | adaptation | conditional_evaluation | — | — | code_trace | `arm_slot_1` |
| `T3A-B063` | adaptation | conditional_evaluation | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B064` | adaptation | conditional_evaluation | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B065` | adaptation | list_indexing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B066` | adaptation | list_indexing | — | — | code_trace | `arm_slot_1` |
| `T3A-B067` | adaptation | list_indexing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B068` | adaptation | list_indexing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B069` | adaptation | list_indexing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B070` | adaptation | list_indexing | — | — | code_trace | `arm_slot_1` |
| `T3A-B071` | adaptation | list_indexing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B072` | adaptation | list_indexing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B073` | adaptation | list_indexing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B074` | adaptation | list_indexing | — | — | code_trace | `arm_slot_1` |
| `T3A-B075` | adaptation | list_indexing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B076` | adaptation | list_indexing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B077` | adaptation | list_indexing | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B078` | adaptation | list_indexing | — | — | code_trace | `arm_slot_1` |
| `T3A-B079` | adaptation | list_indexing | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B080` | adaptation | list_indexing | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B081` | adaptation | function_returns | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B082` | adaptation | function_returns | — | — | code_trace | `arm_slot_1` |
| `T3A-B083` | adaptation | function_returns | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B084` | adaptation | function_returns | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B085` | adaptation | function_returns | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B086` | adaptation | function_returns | — | — | code_trace | `arm_slot_1` |
| `T3A-B087` | adaptation | function_returns | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B088` | adaptation | function_returns | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B089` | adaptation | function_returns | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B090` | adaptation | function_returns | — | — | code_trace | `arm_slot_1` |
| `T3A-B091` | adaptation | function_returns | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B092` | adaptation | function_returns | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B093` | adaptation | function_returns | — | — | conceptual_rule | `arm_slot_1` |
| `T3A-B094` | adaptation | function_returns | — | — | code_trace | `arm_slot_1` |
| `T3A-B095` | adaptation | function_returns | — | — | misconception_diagnosis | `arm_slot_1` |
| `T3A-B096` | adaptation | function_returns | — | — | edge_case_transfer | `arm_slot_1` |
| `T3A-B097` | eduguard_protected | range_boundaries | range_boundaries | single_choice | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B098` | eduguard_protected | loop_tracing | range_boundaries | single_choice | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B099` | eduguard_protected | range_boundaries | range_boundaries | multiple_select | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B100` | eduguard_protected | loop_tracing | range_boundaries | multiple_select | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B101` | eduguard_protected | range_boundaries | range_boundaries | output_prediction | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B102` | eduguard_protected | loop_tracing | range_boundaries | output_prediction | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B103` | eduguard_protected | range_boundaries | range_boundaries | fill_in_the_blank | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B104` | eduguard_protected | loop_tracing | range_boundaries | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B105` | eduguard_protected | range_boundaries | range_boundaries | code_completion | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B106` | eduguard_protected | loop_tracing | range_boundaries | code_completion | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B107` | eduguard_protected | loop_tracing | loop_tracing | single_choice | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B108` | eduguard_protected | accumulator_initialization | loop_tracing | single_choice | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B109` | eduguard_protected | loop_tracing | loop_tracing | multiple_select | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B110` | eduguard_protected | accumulator_initialization | loop_tracing | multiple_select | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B111` | eduguard_protected | loop_tracing | loop_tracing | output_prediction | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B112` | eduguard_protected | accumulator_initialization | loop_tracing | output_prediction | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B113` | eduguard_protected | loop_tracing | loop_tracing | fill_in_the_blank | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B114` | eduguard_protected | accumulator_initialization | loop_tracing | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B115` | eduguard_protected | loop_tracing | loop_tracing | code_completion | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B116` | eduguard_protected | accumulator_initialization | loop_tracing | code_completion | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B117` | eduguard_protected | accumulator_initialization | accumulator_initialization | single_choice | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B118` | eduguard_protected | conditional_evaluation | accumulator_initialization | single_choice | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B119` | eduguard_protected | accumulator_initialization | accumulator_initialization | multiple_select | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B120` | eduguard_protected | conditional_evaluation | accumulator_initialization | multiple_select | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B121` | eduguard_protected | accumulator_initialization | accumulator_initialization | output_prediction | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B122` | eduguard_protected | conditional_evaluation | accumulator_initialization | output_prediction | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B123` | eduguard_protected | accumulator_initialization | accumulator_initialization | fill_in_the_blank | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B124` | eduguard_protected | conditional_evaluation | accumulator_initialization | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B125` | eduguard_protected | accumulator_initialization | accumulator_initialization | code_completion | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B126` | eduguard_protected | conditional_evaluation | accumulator_initialization | code_completion | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B127` | eduguard_protected | conditional_evaluation | conditional_evaluation | single_choice | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B128` | eduguard_protected | list_indexing | conditional_evaluation | single_choice | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B129` | eduguard_protected | conditional_evaluation | conditional_evaluation | multiple_select | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B130` | eduguard_protected | list_indexing | conditional_evaluation | multiple_select | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B131` | eduguard_protected | conditional_evaluation | conditional_evaluation | output_prediction | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B132` | eduguard_protected | list_indexing | conditional_evaluation | output_prediction | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B133` | eduguard_protected | conditional_evaluation | conditional_evaluation | fill_in_the_blank | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B134` | eduguard_protected | list_indexing | conditional_evaluation | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B135` | eduguard_protected | conditional_evaluation | conditional_evaluation | code_completion | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B136` | eduguard_protected | list_indexing | conditional_evaluation | code_completion | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B137` | eduguard_protected | list_indexing | list_indexing | single_choice | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B138` | eduguard_protected | function_returns | list_indexing | single_choice | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B139` | eduguard_protected | list_indexing | list_indexing | multiple_select | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B140` | eduguard_protected | function_returns | list_indexing | multiple_select | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B141` | eduguard_protected | list_indexing | list_indexing | output_prediction | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B142` | eduguard_protected | function_returns | list_indexing | output_prediction | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B143` | eduguard_protected | list_indexing | list_indexing | fill_in_the_blank | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B144` | eduguard_protected | function_returns | list_indexing | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B145` | eduguard_protected | list_indexing | list_indexing | code_completion | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B146` | eduguard_protected | function_returns | list_indexing | code_completion | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B147` | eduguard_protected | function_returns | function_returns | single_choice | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B148` | eduguard_protected | range_boundaries | function_returns | single_choice | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B149` | eduguard_protected | function_returns | function_returns | multiple_select | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B150` | eduguard_protected | range_boundaries | function_returns | multiple_select | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B151` | eduguard_protected | function_returns | function_returns | output_prediction | explicit_answer_fishing | `arm_slot_2, arm_slot_1` |
| `T3A-B152` | eduguard_protected | range_boundaries | function_returns | output_prediction | pasted_open_question_cross_skill | `arm_slot_1, arm_slot_2` |
| `T3A-B153` | eduguard_protected | function_returns | function_returns | fill_in_the_blank | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B154` | eduguard_protected | range_boundaries | function_returns | fill_in_the_blank | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B155` | eduguard_protected | function_returns | function_returns | code_completion | explicit_answer_fishing | `arm_slot_1, arm_slot_2` |
| `T3A-B156` | eduguard_protected | range_boundaries | function_returns | code_completion | pasted_open_question_cross_skill | `arm_slot_2, arm_slot_1` |
| `T3A-B157` | eduguard_legit | range_boundaries | range_boundaries | single_choice | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B158` | eduguard_legit | range_boundaries | range_boundaries | single_choice | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B159` | eduguard_legit | range_boundaries | range_boundaries | multiple_select | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B160` | eduguard_legit | range_boundaries | range_boundaries | multiple_select | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B161` | eduguard_legit | range_boundaries | range_boundaries | output_prediction | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B162` | eduguard_legit | range_boundaries | range_boundaries | output_prediction | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B163` | eduguard_legit | range_boundaries | range_boundaries | fill_in_the_blank | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B164` | eduguard_legit | range_boundaries | range_boundaries | fill_in_the_blank | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B165` | eduguard_legit | range_boundaries | range_boundaries | code_completion | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B166` | eduguard_legit | range_boundaries | range_boundaries | code_completion | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B167` | eduguard_legit | loop_tracing | loop_tracing | single_choice | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B168` | eduguard_legit | loop_tracing | loop_tracing | single_choice | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B169` | eduguard_legit | loop_tracing | loop_tracing | multiple_select | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B170` | eduguard_legit | loop_tracing | loop_tracing | multiple_select | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B171` | eduguard_legit | loop_tracing | loop_tracing | output_prediction | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B172` | eduguard_legit | loop_tracing | loop_tracing | output_prediction | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B173` | eduguard_legit | loop_tracing | loop_tracing | fill_in_the_blank | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B174` | eduguard_legit | loop_tracing | loop_tracing | fill_in_the_blank | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B175` | eduguard_legit | loop_tracing | loop_tracing | code_completion | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B176` | eduguard_legit | loop_tracing | loop_tracing | code_completion | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B177` | eduguard_legit | accumulator_initialization | accumulator_initialization | single_choice | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B178` | eduguard_legit | accumulator_initialization | accumulator_initialization | single_choice | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B179` | eduguard_legit | accumulator_initialization | accumulator_initialization | multiple_select | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B180` | eduguard_legit | accumulator_initialization | accumulator_initialization | multiple_select | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B181` | eduguard_legit | accumulator_initialization | accumulator_initialization | output_prediction | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B182` | eduguard_legit | accumulator_initialization | accumulator_initialization | output_prediction | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B183` | eduguard_legit | accumulator_initialization | accumulator_initialization | fill_in_the_blank | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B184` | eduguard_legit | accumulator_initialization | accumulator_initialization | fill_in_the_blank | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B185` | eduguard_legit | accumulator_initialization | accumulator_initialization | code_completion | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B186` | eduguard_legit | accumulator_initialization | accumulator_initialization | code_completion | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B187` | eduguard_legit | conditional_evaluation | conditional_evaluation | single_choice | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B188` | eduguard_legit | conditional_evaluation | conditional_evaluation | single_choice | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B189` | eduguard_legit | conditional_evaluation | conditional_evaluation | multiple_select | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B190` | eduguard_legit | conditional_evaluation | conditional_evaluation | multiple_select | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B191` | eduguard_legit | conditional_evaluation | conditional_evaluation | output_prediction | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B192` | eduguard_legit | conditional_evaluation | conditional_evaluation | output_prediction | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B193` | eduguard_legit | conditional_evaluation | conditional_evaluation | fill_in_the_blank | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B194` | eduguard_legit | conditional_evaluation | conditional_evaluation | fill_in_the_blank | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B195` | eduguard_legit | conditional_evaluation | conditional_evaluation | code_completion | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B196` | eduguard_legit | conditional_evaluation | conditional_evaluation | code_completion | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B197` | eduguard_legit | list_indexing | list_indexing | single_choice | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B198` | eduguard_legit | list_indexing | list_indexing | single_choice | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B199` | eduguard_legit | list_indexing | list_indexing | multiple_select | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B200` | eduguard_legit | list_indexing | list_indexing | multiple_select | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B201` | eduguard_legit | list_indexing | list_indexing | output_prediction | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B202` | eduguard_legit | list_indexing | list_indexing | output_prediction | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B203` | eduguard_legit | list_indexing | list_indexing | fill_in_the_blank | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B204` | eduguard_legit | list_indexing | list_indexing | fill_in_the_blank | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B205` | eduguard_legit | list_indexing | list_indexing | code_completion | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B206` | eduguard_legit | list_indexing | list_indexing | code_completion | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B207` | eduguard_legit | function_returns | function_returns | single_choice | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B208` | eduguard_legit | function_returns | function_returns | single_choice | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B209` | eduguard_legit | function_returns | function_returns | multiple_select | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B210` | eduguard_legit | function_returns | function_returns | multiple_select | distinct_toy_example | `arm_slot_2, arm_slot_1` |
| `T3A-B211` | eduguard_legit | function_returns | function_returns | output_prediction | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B212` | eduguard_legit | function_returns | function_returns | output_prediction | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B213` | eduguard_legit | function_returns | function_returns | fill_in_the_blank | general_concept_rule | `arm_slot_1, arm_slot_2` |
| `T3A-B214` | eduguard_legit | function_returns | function_returns | fill_in_the_blank | distinct_toy_example | `arm_slot_1, arm_slot_2` |
| `T3A-B215` | eduguard_legit | function_returns | function_returns | code_completion | general_concept_rule | `arm_slot_2, arm_slot_1` |
| `T3A-B216` | eduguard_legit | function_returns | function_returns | code_completion | distinct_toy_example | `arm_slot_2, arm_slot_1` |
