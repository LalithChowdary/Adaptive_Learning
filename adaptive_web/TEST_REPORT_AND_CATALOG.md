# Complete Verification & Test Catalog (`3/adaptive_web/` & Regression Suites)

This document records all **246 automated tests** executed and verified (`201/201` in `3/adaptive_web/tests/` plus `45/45` regression tests in `3/dashboard/`, `3/ppo/`, and `3/model/`).

---

## 1. Summary Table

| Suite / Tier | File Path | Tests | Result | Scope |
| :--- | :--- | :---: | :---: | :--- |
| **Unit Suite 1 — Core Engine & Adapters** | `3/adaptive_web/tests/test_core.py` | **22** | `PASS (22/22)` | Automated verification |
| **Unit Suite 2 — LLM Pedagogy Realizer** | `3/adaptive_web/tests/test_llm_pedagogy.py` | **6** | `PASS (6/6)` | Automated verification |
| **E2E Tier 0 — Infrastructure & Harness Smoke** | `3/adaptive_web/tests/e2e/test_tier0_infra_smoke.py` | **4** | `PASS (4/4)` | Automated verification |
| **E2E Tier 1 — Category-Partition Feature Matrix F01..F14** | `3/adaptive_web/tests/e2e/test_tier1_features.py` | **73** | `PASS (73/73)` | Automated verification |
| **E2E Tier 2 — Boundary, Security & Negative Tests F01..F14** | `3/adaptive_web/tests/e2e/test_tier2_boundaries.py` | **72** | `PASS (72/72)` | Automated verification |
| **E2E Tier 3 — Pairwise Cross-Component Integration** | `3/adaptive_web/tests/e2e/test_tier3_pairwise.py` | **16** | `PASS (16/16)` | Automated verification |
| **E2E Tier 4 — End-to-End Pedagogical & Presenter Scenarios** | `3/adaptive_web/tests/e2e/test_tier4_scenarios.py` | **8** | `PASS (8/8)` | Automated verification |
| **Regression Suite 1 — Reference Dashboard** | `3/dashboard/tests/test_dashboard.py` | **26** | `PASS (26/26)` | Automated verification |
| **Regression Suite 2 — Frozen PPO Policy & Env** | `3/ppo/tests/test_ppo.py` | **6** | `PASS (6/6)` | Automated verification |
| **Regression Suite 3 — Frozen Monotonic GRU Memory** | `3/model/tests/test_memory.py` | **13** | `PASS (13/13)` | Automated verification |
| **Adaptive Web Subtotal** | `3/adaptive_web/tests/` | **201** | **PASS (201/201)** | Unit + Tier 0–4 E2E |
| **Grand Total (with Regressions)** | All Suites | **246** | **PASS (246/246)** | Full repository verification |

---

## 2. How to Run All Tests

```bash
# Run all 201 adaptive_web tests (28 unit + 173 E2E)
3/model/.venv/bin/python -m unittest discover -s 3/adaptive_web/tests -p "test_*.py" -v

# Run only the 173 opaque-box E2E tests (Tiers 0..4)
3/model/.venv/bin/python -m unittest discover -s 3/adaptive_web/tests/e2e -p "test_*.py" -v

# Run existing regression suites (45 tests)
3/model/.venv/bin/python -m unittest discover -s 3/dashboard/tests -v
PYTHONPATH=3 3/model/.venv/bin/python -m unittest discover -s 3/ppo/tests -v
PYTHONPATH=3/model 3/model/.venv/bin/python -m unittest discover -s 3/model/tests -v
```

---

## 3. Complete Itemized Catalog of All 246 Tests

### Unit Suite 1 — Core Engine & Adapters (`3/adaptive_web/tests/test_core.py`) (22 tests)

1. `ConfigTests.test_defaults_without_env_file` — **PASS**
2. `ConfigTests.test_dotenv_parsing_and_os_environ_precedence` — **PASS**
3. `ConfigTests.test_provider_key_fallback_and_validation` — **PASS**
4. `StoreTests.test_wal_mode_tables_and_profiles` — **PASS**
5. `StoreTests.test_get_task_v22_and_generated_tasks_and_isolation` — **PASS**
6. `StoreTests.test_idempotency_nested_rollback_and_concurrency` — **PASS**
7. `CatalogueBridgeTests.test_v22_2160_items_and_ppo_skill_order` — **PASS**
8. `CatalogueBridgeTests.test_8_key_memory_features_match_dashboard_across_all_2160` — **PASS**
9. `CatalogueBridgeTests.test_validate_task_schema_valid_and_invalid_all_5_formats` — **PASS**
10. `MemoryAdapterTests.test_checkpoint_contract_and_no_dashboard_import` — **PASS**
11. `MemoryAdapterTests.test_cold_start_unassessed_and_observed_vs_predicted_separation` — **PASS**
12. `MemoryAdapterTests.test_learner_skill_isolation_and_rollback_cache_safety` — **PASS**
13. `MemoryAdapterTests.test_assisted_only_has_none_unaided_pass_rate_and_curves_monotonic` — **PASS**
14. `MemoryAdapterTests.test_inspect_tokens_47_dim_zero_text_embeddings` — **PASS**
15. `MemoryAdapterTests.test_equivalence_with_dashboard_memory_service` — **PASS**
16. `PPOEngineTests.test_obs_vector_36_bounds_groups_and_skill_order` — **PASS**
17. `PPOEngineTests.test_gated_option_b_masks_and_reasons` — **PASS**
18. `PPOEngineTests.test_generalized_reconstruct_context_v22_and_generated_tasks` — **PASS**
19. `PPOEngineTests.test_3_seed_ensemble_probs_critic_and_rationales` — **PASS**
20. `GraderTests.test_all_5_formats_deterministic` — **PASS**
21. `GraderTests.test_ast_interpreter_pass_fail_syntax_and_unresolved` — **PASS**
22. `GraderTests.test_grade_self_explanation_separates_confidence_from_score` — **PASS**

### Unit Suite 2 — LLM Pedagogy Realizer (`3/adaptive_web/tests/test_llm_pedagogy.py`) (6 tests)

1. `PedagogyRealizerTests.test_curate_learning_path_unassessed_vs_fragile` — **PASS**
2. `PedagogyRealizerTests.test_generate_course_material_structure` — **PASS**
3. `PedagogyRealizerTests.test_generate_question_valid_schema` — **PASS**
4. `PedagogyRealizerTests.test_generate_hints_progressive_leakage` — **PASS**
5. `PedagogyRealizerTests.test_evaluate_self_explanation_scoring` — **PASS**
6. `PedagogyRealizerTests.test_realize_ppo_turns` — **PASS**

### E2E Tier 0 — Infrastructure & Harness Smoke (`3/adaptive_web/tests/e2e/test_tier0_infra_smoke.py`) (4 tests)

1. `TestTier0InfraSmoke.test_tier0_01_check_counts_gate_passes` — **PASS**
2. `TestTier0InfraSmoke.test_tier0_02_unittest_discover_collects_at_least_172_with_zero_import_errors` — **PASS**
3. `TestTier0InfraSmoke.test_tier0_03_local_llm_stub_loopback_delay_error_and_clean_teardown` — **PASS**
4. `TestTier0InfraSmoke.test_tier0_04_v22_catalog_resolver_all_5_formats_and_reference_hashes` — **PASS**

### E2E Tier 1 — Category-Partition Feature Matrix F01..F14 (`3/adaptive_web/tests/e2e/test_tier1_features.py`) (73 tests)

1. `TestF01Isolation.test_f01_default_port_8780_and_separate_db_path` — **PASS**
2. `TestF01Isolation.test_f01_reference_artifacts_unchanged_after_sessions` — **PASS**
3. `TestF01Isolation.test_f01_cli_server_startup_from_neutral_cwd` — **PASS**
4. `TestF01Isolation.test_f01_dashboard_sqlite_untouched_during_e2e_writes` — **PASS**
5. `TestF01Isolation.test_f01_static_routes_serve_adaptive_web_not_dashboard` — **PASS**
6. `TestF02ConfigEnv.test_f02_env_var_overrides_port_db_device_provider` — **PASS**
7. `TestF02ConfigEnv.test_f02_dotenv_file_loaded_without_external_dotenv` — **PASS**
8. `TestF02ConfigEnv.test_f02_os_environ_precedence_over_dotenv` — **PASS**
9. `TestF02ConfigEnv.test_f02_provider_specific_api_keys_auto_detected` — **PASS**
10. `TestF02ConfigEnv.test_f02_api_keys_never_exposed_in_http_payloads_or_export` — **PASS**
11. `TestF03StoreIdempotency.test_f03_sqlite_wal_tables_and_persistence_across_restart` — **PASS**
12. `TestF03StoreIdempotency.test_f03_post_idempotency_same_request_id_returns_cached` — **PASS**
13. `TestF03StoreIdempotency.test_f03_profile_isolation_across_two_learners` — **PASS**
14. `TestF03StoreIdempotency.test_f03_skill_isolation_within_same_learner` — **PASS**
15. `TestF03StoreIdempotency.test_f03_export_schema_contains_all_provenance_artifacts` — **PASS**
16. `TestF04BankDAG.test_f04_all_six_canonical_skills_in_ppo_order_and_dag` — **PASS**
17. `TestF04BankDAG.test_f04_all_five_task_formats_deliverable_from_v22` — **PASS**
18. `TestF04BankDAG.test_f04_practice_vs_assessment_pool_routing` — **PASS**
19. `TestF04BankDAG.test_f04_unseen_item_selection_avoids_repeats` — **PASS**
20. `TestF04BankDAG.test_f04_v22_task_answer_key_and_criteria_hidden_before_submit` — **PASS**
21. `TestF05MemoryAdapter.test_f05_unassessed_cold_start_guard_and_priors` — **PASS**
22. `TestF05MemoryAdapter.test_f05_observed_grades_strictly_separated_from_predictions` — **PASS**
23. `TestF05MemoryAdapter.test_f05_0_to_60_day_retention_curves_monotonic_and_difficulty_ordered` — **PASS**
24. `TestF05MemoryAdapter.test_f05_curve_supports_all_five_task_formats` — **PASS**
25. `TestF05MemoryAdapter.test_f05_structured_token_audit_zero_text_or_code_embeddings` — **PASS**
26. `TestF05MemoryAdapter.test_f05_clean_token_speed_deconfounding_zero_score_invariant` — **PASS**
27. `TestF06PPOEngine.test_f06_36_dim_observation_bounds_and_3seed_ensemble` — **PASS**
28. `TestF06PPOEngine.test_f06_cold_start_valid_mask_actions_1_2_3_4` — **PASS**
29. `TestF06PPOEngine.test_f06_gated_option_b_after_passive_lesson_forces_question` — **PASS**
30. `TestF06PPOEngine.test_f06_self_explanation_unlocks_after_question_and_relocks` — **PASS**
31. `TestF06PPOEngine.test_f06_generalized_reconstruct_context_tracks_llm_generated_tasks` — **PASS**
32. `TestF06PPOEngine.test_f06_manual_override_logs_learner_manual_provenance` — **PASS**
33. `TestF07Grader.test_f07_single_choice_correct_and_wrong_grading` — **PASS**
34. `TestF07Grader.test_f07_multiple_select_partial_and_penalty_grading` — **PASS**
35. `TestF07Grader.test_f07_output_prediction_normalization_and_partial_credit` — **PASS**
36. `TestF07Grader.test_f07_fill_in_the_blank_exact_int_grading` — **PASS**
37. `TestF07Grader.test_f07_code_completion_ast_interpreter_grades_valid_solve` — **PASS**
38. `TestF07Grader.test_f07_self_explanation_grader_separates_confidence_from_score` — **PASS**
39. `TestF08LLMClient.test_f08_mock_provider_deterministic_double_reproducible` — **PASS**
40. `TestF08LLMClient.test_f08_openai_compatible_http_client_via_loopback_stub` — **PASS**
41. `TestF08LLMClient.test_f08_anthropic_http_client_via_loopback_stub` — **PASS**
42. `TestF08LLMClient.test_f08_gemini_http_client_via_loopback_stub` — **PASS**
43. `TestF08LLMClient.test_f08_http_error_or_malformed_json_falls_back_gracefully` — **PASS**
44. `TestF09PathCourse.test_f09_select_skill_generates_curated_path_and_material` — **PASS**
45. `TestF09PathCourse.test_f09_curated_path_reflects_execute_program_cadence` — **PASS**
46. `TestF09PathCourse.test_f09_path_adapts_for_assessed_weak_vs_strong_memory` — **PASS**
47. `TestF09PathCourse.test_f09_worked_examples_include_subgoal_labels` — **PASS**
48. `TestF09PathCourse.test_f09_path_and_material_persisted_per_learner_and_skill` — **PASS**
49. `TestF10LLMQuestions.test_f10_generate_fresh_llm_question_across_formats` — **PASS**
50. `TestF10LLMQuestions.test_f10_llm_question_emits_exact_8key_memory_task_metadata` — **PASS**
51. `TestF10LLMQuestions.test_f10_llm_code_completion_executable_in_ast_interpreter` — **PASS**
52. `TestF10LLMQuestions.test_f10_llm_question_updates_monotonic_gru_identically_to_v22` — **PASS**
53. `TestF10LLMQuestions.test_f10_invalid_llm_question_payload_falls_back_cleanly` — **PASS**
54. `TestF11BoundedPPOTurns.test_f11_all_eight_ppo_actions_realizable_through_api` — **PASS**
55. `TestF11BoundedPPOTurns.test_f11_progressive_3tier_hints_emit_leakage_and_kinds` — **PASS**
56. `TestF11BoundedPPOTurns.test_f11_assisted_pass_sets_unverified_and_triggers_unaided_check` — **PASS**
57. `TestF11BoundedPPOTurns.test_f11_self_explanation_submission_consolidates_permastore_floor` — **PASS**
58. `TestF11BoundedPPOTurns.test_f11_retry_and_reveal_emit_clean_events_and_close_correctly` — **PASS**
59. `TestF12HTTPAPI.test_f12_all_rest_endpoints_happy_path_status_and_json` — **PASS**
60. `TestF12HTTPAPI.test_f12_security_headers_csp_nosniff_and_no_store` — **PASS**
61. `TestF12HTTPAPI.test_f12_non_blocking_reads_during_slow_outbound_llm_call` — **PASS**
62. `TestF12HTTPAPI.test_f12_glass_box_telemetry_fields_complete_in_api_state` — **PASS**
63. `TestF12HTTPAPI.test_f12_large_code_or_course_json_post_accepted_cleanly` — **PASS**
64. `TestF13FrontendUX.test_f13_css_matches_exact_dashboard_palette_tokens` — **PASS**
65. `TestF13FrontendUX.test_f13_css_has_zero_purple_or_linear_radial_gradients` — **PASS**
66. `TestF13FrontendUX.test_f13_html_and_js_expose_dag_workspace_and_glass_box_toggle` — **PASS**
67. `TestF13FrontendUX.test_f13_frontend_js_renders_0_to_60d_svg_and_unassessed_guard` — **PASS**
68. `TestF13FrontendUX.test_f13_active_workspace_state_persists_across_reload_bootstrap` — **PASS**
69. `TestF14ProposalDoc.test_f14_proposal_file_exists_and_has_structured_sections` — **PASS**
70. `TestF14ProposalDoc.test_f14_proposal_compares_background_suggestion_vs_hard_ppo_control` — **PASS**
71. `TestF14ProposalDoc.test_f14_proposal_grounds_ux_in_all_five_research_systems` — **PASS**
72. `TestF14ProposalDoc.test_f14_proposal_documents_glass_box_inspector_and_provenance` — **PASS**
73. `TestF14ProposalDoc.test_f14_proposal_matches_exact_model_and_ppo_dimensions` — **PASS**

### E2E Tier 2 — Boundary, Security & Negative Tests F01..F14 (`3/adaptive_web/tests/e2e/test_tier2_boundaries.py`) (72 tests)

1. `TestF01Boundary.test_f01_b_static_path_traversal_rejected_with_404_or_403` — **PASS**
2. `TestF01Boundary.test_f01_b_missing_frozen_checkpoint_path_fails_explicitly` — **PASS**
3. `TestF01Boundary.test_f01_b_read_only_db_directory_fails_cleanly_on_startup` — **PASS**
4. `TestF01Boundary.test_f01_b_port_collision_raises_oserror_cleanly` — **PASS**
5. `TestF01Boundary.test_f01_b_unrecognized_http_methods_put_delete_rejected` — **PASS**
6. `TestF02Boundary.test_f02_b_empty_api_key_whitespace_falls_back_to_mock` — **PASS**
7. `TestF02Boundary.test_f02_b_invalid_port_out_of_range_rejected` — **PASS**
8. `TestF02Boundary.test_f02_b_invalid_llm_provider_or_device_rejected` — **PASS**
9. `TestF02Boundary.test_f02_b_timeout_zero_or_negative_rejected` — **PASS**
10. `TestF02Boundary.test_f02_b_dotenv_malformed_lines_and_equals_in_value` — **PASS**
11. `TestF03Boundary.test_f03_b_request_id_length_boundaries_7_8_100_101` — **PASS**
12. `TestF03Boundary.test_f03_b_profile_name_length_boundaries_0_whitespace_1_50_51` — **PASS**
13. `TestF03Boundary.test_f03_b_unknown_learner_id_returns_400_on_all_endpoints` — **PASS**
14. `TestF03Boundary.test_f03_b_duplicate_submit_with_new_request_id_rejected` — **PASS**
15. `TestF03Boundary.test_f03_b_failed_transaction_rolls_back_db_and_memory_cache` — **PASS**
16. `TestF04Boundary.test_f04_b_unknown_skill_id_rejected_across_endpoints` — **PASS**
17. `TestF04Boundary.test_f04_b_unknown_format_or_difficulty_rejected` — **PASS**
18. `TestF04Boundary.test_f04_b_root_skill_range_boundaries_has_empty_prerequisites` — **PASS**
19. `TestF04Boundary.test_f04_b_bank_exhaustion_in_single_filter_cell_handled_cleanly` — **PASS**
20. `TestF04Boundary.test_f04_b_curve_rejects_mismatched_skill_and_task_id` — **PASS**
21. `TestF05Boundary.test_f05_b_zero_score_fast_guess_decreases_prediction_not_increases` — **PASS**
22. `TestF05Boundary.test_f05_b_unresolved_ast_grade_does_not_penalize_like_wrong_answer` — **PASS**
23. `TestF05Boundary.test_f05_b_non_evidence_events_keep_skill_unassessed` — **PASS**
24. `TestF05Boundary.test_f05_b_day_60_curve_boundary_and_floor_clamping` — **PASS**
25. `TestF05Boundary.test_f05_b_repeated_read_queries_idempotent_on_state` — **PASS**
26. `TestF05Boundary.test_f05_b_unaided_vs_assisted_3hint_success_ordering` — **PASS**
27. `TestF06Boundary.test_f06_b_masked_action_invocation_rejected_with_400` — **PASS**
28. `TestF06Boundary.test_f06_b_unknown_action_id_minus1_and_8_rejected` — **PASS**
29. `TestF06Boundary.test_f06_b_budget_wrap_at_total_budget_32_boundary` — **PASS**
30. `TestF06Boundary.test_f06_b_awaiting_question_is_scoped_per_skill` — **PASS**
31. `TestF06Boundary.test_f06_b_review_prerequisite_unmasked_at_threshold_and_masked_when_healthy` — **PASS**
32. `TestF07Boundary.test_f07_b_multiple_select_empty_and_all_selected_clamp_to_zero` — **PASS**
33. `TestF07Boundary.test_f07_b_code_completion_syntax_error_is_resolved_zero` — **PASS**
34. `TestF07Boundary.test_f07_b_code_completion_infinite_loop_and_disallowed_import_unresolved` — **PASS**
35. `TestF07Boundary.test_f07_b_code_completion_exceeding_8000_chars_or_1200_ast_nodes_unresolved` — **PASS**
36. `TestF07Boundary.test_f07_b_fill_in_the_blank_rejects_float_or_non_int_string` — **PASS**
37. `TestF07Boundary.test_f07_b_active_seconds_boundary_zero_negative_nan_and_huge` — **PASS**
38. `TestF08Boundary.test_f08_b_stub_timeout_triggers_clean_fallback` — **PASS**
39. `TestF08Boundary.test_f08_b_stub_http_429_rate_limit_fallback` — **PASS**
40. `TestF08Boundary.test_f08_b_stub_truncated_json_and_missing_keys_fallback` — **PASS**
41. `TestF08Boundary.test_f08_b_markdown_fenced_json_from_llm_parsed_cleanly` — **PASS**
42. `TestF08Boundary.test_f08_b_auto_provider_without_keys_never_makes_network_call` — **PASS**
43. `TestF09Boundary.test_f09_b_repeated_select_skill_same_skill_clean_no_duplicate_crash` — **PASS**
44. `TestF09Boundary.test_f09_b_select_skill_with_active_open_question_preserves_active_item` — **PASS**
45. `TestF09Boundary.test_f09_b_course_material_escapes_html_script_tags_safely` — **PASS**
46. `TestF09Boundary.test_f09_b_all_six_skills_generate_distinct_skill_specific_materials` — **PASS**
47. `TestF09Boundary.test_f09_b_empty_focus_gap_handled_in_course_material` — **PASS**
48. `TestF10Boundary.test_f10_b_llm_question_credit_weights_sum_to_one_tolerance` — **PASS**
49. `TestF10Boundary.test_f10_b_llm_question_code_completion_rejects_disallowed_builtins` — **PASS**
50. `TestF10Boundary.test_f10_b_llm_question_multiple_select_penalty_weights_non_empty` — **PASS**
51. `TestF10Boundary.test_f10_b_llm_question_duplicate_or_empty_options_rejected_with_fallback` — **PASS**
52. `TestF10Boundary.test_f10_b_llm_question_zero_or_negative_reference_seconds_rejected` — **PASS**
53. `TestF11Boundary.test_f11_b_hint_cap_at_three_rejects_fourth_request` — **PASS**
54. `TestF11Boundary.test_f11_b_hint_after_correct_submission_rejected` — **PASS**
55. `TestF11Boundary.test_f11_b_retry_cap_at_two_rejects_third_retry` — **PASS**
56. `TestF11Boundary.test_f11_b_reveal_before_first_submission_rejected` — **PASS**
57. `TestF11Boundary.test_f11_b_empty_self_explanation_submission_graded_zero_not_500` — **PASS**
58. `TestF12Boundary.test_f12_b_missing_or_wrong_csrf_token_rejected_with_403` — **PASS**
59. `TestF12Boundary.test_f12_b_disallowed_origin_and_host_headers_rejected_with_403` — **PASS**
60. `TestF12Boundary.test_f12_b_non_json_content_type_and_malformed_json_rejected` — **PASS**
61. `TestF12Boundary.test_f12_b_oversized_body_and_zero_length_post_rejected` — **PASS**
62. `TestF12Boundary.test_f12_b_nan_infinity_rejected_in_json_submit` — **PASS**
63. `TestF13Boundary.test_f13_b_css_zero_forbidden_purple_gradient_hex_or_css_functions` — **PASS**
64. `TestF13Boundary.test_f13_b_css_includes_focus_visible_accessibility_outline` — **PASS**
65. `TestF13Boundary.test_f13_b_css_responsive_breakpoints_and_reduced_motion` — **PASS**
66. `TestF13Boundary.test_f13_b_favicon_svg_served_valid_xml_with_brand_palette` — **PASS**
67. `TestF13Boundary.test_f13_b_api_state_without_learner_still_boots_cleanly` — **PASS**
68. `TestF14Boundary.test_f14_b_proposal_contains_no_unresolved_placeholders` — **PASS**
69. `TestF14Boundary.test_f14_b_proposal_contains_zero_secrets_or_api_keys` — **PASS**
70. `TestF14Boundary.test_f14_b_proposal_discloses_simulated_calibration_limitation` — **PASS**
71. `TestF14Boundary.test_f14_b_proposal_referenced_local_paths_exist` — **PASS**
72. `TestF14Boundary.test_f14_b_proposal_port_and_palette_claims_consistent` — **PASS**

### E2E Tier 3 — Pairwise Cross-Component Integration (`3/adaptive_web/tests/e2e/test_tier3_pairwise.py`) (16 tests)

1. `TestTier3Pairwise.test_pair_01_memory_decay_updates_ppo_s_t_and_action_probs` — **PASS**
2. `TestTier3Pairwise.test_pair_02_ppo_step_with_llm_question_and_subsequent_mask` — **PASS**
3. `TestTier3Pairwise.test_pair_03_ast_grader_evaluates_llm_code_completion_variants` — **PASS**
4. `TestTier3Pairwise.test_pair_04_llm_hints_update_leakage_l_t_and_unverified_flag` — **PASS**
5. `TestTier3Pairwise.test_pair_05_concurrent_duplicate_request_id_posts_safe` — **PASS**
6. `TestTier3Pairwise.test_pair_06_dag_prerequisites_surface_in_curated_path` — **PASS**
7. `TestTier3Pairwise.test_pair_07_unassessed_skill_guard_across_state_curve_and_export` — **PASS**
8. `TestTier3Pairwise.test_pair_08_unresolved_submission_preserves_observed_vs_predicted_integrity` — **PASS**
9. `TestTier3Pairwise.test_pair_09_manual_override_vs_ppo_provenance_in_export` — **PASS**
10. `TestTier3Pairwise.test_pair_10_env_config_switches_stub_provider_cleanly` — **PASS**
11. `TestTier3Pairwise.test_pair_11_self_explanation_consolidates_b_t_more_than_passive_lesson` — **PASS**
12. `TestTier3Pairwise.test_pair_12_slow_llm_path_generation_never_blocks_v22_submit` — **PASS**
13. `TestTier3Pairwise.test_pair_13_two_sequential_harnesses_never_share_sqlite_state` — **PASS**
14. `TestTier3Pairwise.test_pair_14_static_assets_and_api_share_consistent_security_and_palette` — **PASS**
15. `TestTier3Pairwise.test_pair_15_action_3_4_primed_lesson_links_with_course_material` — **PASS**
16. `TestTier3Pairwise.test_pair_16_proposal_documented_actions_and_endpoints_match_live_api` — **PASS**

### E2E Tier 4 — End-to-End Pedagogical & Presenter Scenarios (`3/adaptive_web/tests/e2e/test_tier4_scenarios.py`) (8 tests)

1. `TestTier4Scenarios.test_scenario_01_cold_start_beginner_full_learn_code_remember_loop` — **PASS**
2. `TestTier4Scenarios.test_scenario_02_struggling_learner_3hints_assisted_pass_and_unaided_verification` — **PASS**
3. `TestTier4Scenarios.test_scenario_03_prerequisite_weakness_and_decay_detour_action_6` — **PASS**
4. `TestTier4Scenarios.test_scenario_04_hybrid_v22_and_fresh_llm_5format_mastery_with_ast_code` — **PASS**
5. `TestTier4Scenarios.test_scenario_05_assessment_integrity_help_before_submit_and_immutable_first_grade` — **PASS**
6. `TestTier4Scenarios.test_scenario_06_server_restart_persistence_and_multi_learner_isolation` — **PASS**
7. `TestTier4Scenarios.test_scenario_07_presenter_showcase_glass_box_audit_manual_override_and_palette` — **PASS**
8. `TestTier4Scenarios.test_scenario_08_concurrent_learners_with_slow_loopback_llm_stub` — **PASS**

### Regression Suite 1 — Reference Dashboard (`3/dashboard/tests/test_dashboard.py`) (26 tests)

1. `GradingTests.test_all_2160_reference_answers` — **PASS**
2. `GradingTests.test_catalogue_structure` — **PASS**
3. `GradingTests.test_multiple_select_penalties` — **PASS**
4. `GradingTests.test_open_partial_and_format_normalization` — **PASS**
5. `GradingTests.test_code_alternative_solution_and_partial` — **PASS**
6. `GradingTests.test_unsupported_never_marked_wrong` — **PASS**
7. `GradingTests.test_syntax_error_is_resolved_failure` — **PASS**
8. `AppTests.test_prediction_hidden_and_no_answer_leak` — **PASS**
9. `AppTests.test_submit_hint_retry_and_fresh_target` — **PASS**
10. `AppTests.test_assessment_help_before_submission_incomplete` — **PASS**
11. `AppTests.test_saved_assessment_not_overwritten` — **PASS**
12. `AppTests.test_retry_cap` — **PASS**
13. `AppTests.test_idempotent_request_and_duplicate_submit` — **PASS**
14. `AppTests.test_restart_and_profile_skill_isolation` — **PASS**
15. `AppTests.test_queries_do_not_change_events_or_state` — **PASS**
16. `AppTests.test_feature_contract_matches_dataset_projection` — **PASS**
17. `AppTests.test_replay_matches_original_model_queries` — **PASS**
18. `AppTests.test_fast_wrong_answer_decreases_prediction_and_curve_non_increasing` — **PASS**
19. `AppTests.test_two_hard_successes_retain_realistic_long_horizon_curve` — **PASS**
20. `AppTests.test_export_no_synthetic_events` — **PASS**
21. `AppTests.test_http_security_and_static` — **PASS**
22. `AppTests.test_ppo_state_observation_and_initial_mask` — **PASS**
23. `AppTests.test_ppo_step_lesson_gated_option_b_and_followup_question` — **PASS**
24. `AppTests.test_ppo_self_explanation_unlocks_after_question_and_locks_after_use` — **PASS**
25. `AppTests.test_ppo_assisted_hint_sets_unverified_and_prereq_or_advance` — **PASS**
26. `AppTests.test_curve_format_selector_and_difficulty_monotonicity` — **PASS**

### Regression Suite 2 — Frozen PPO Policy & Env (`3/ppo/tests/test_ppo.py`) (6 tests)

1. `PPOTests.test_reset_observation_bounds_and_initial_mask` — **PASS**
2. `PPOTests.test_gated_option_b_prevents_consecutive_passive_lectures` — **PASS**
3. `PPOTests.test_hint_leakage_sets_and_unaided_check_clears_unverified_flag` — **PASS**
4. `PPOTests.test_self_explanation_consolidates_floor_and_does_not_leak_delayed_clones` — **PASS**
5. `PPOTests.test_masked_actor_critic_never_selects_invalid_actions_and_updates` — **PASS**
6. `PPOTests.test_baseline_policies_complete_full_episode` — **PASS**

### Regression Suite 3 — Frozen Monotonic GRU Memory (`3/model/tests/test_memory.py`) (13 tests)

1. `EncodingTests.test_causal_pairing_and_elapsed_time_through_skipped_events` — **PASS**
2. `EncodingTests.test_future_outcome_changes_neither_early_inputs_nor_labels` — **PASS**
3. `EncodingTests.test_unknown_grade_is_distinct_from_failure` — **PASS**
4. `EncodingTests.test_mismatched_grade_fails_instead_of_guessing` — **PASS**
5. `EncodingTests.test_identifiers_do_not_enter_features` — **PASS**
6. `EncodingTests.test_training_rejects_assessment_queries` — **PASS**
7. `EncodingTests.test_v21_training_probes_share_the_frozen_practice_state` — **PASS**
8. `EvaluationTests.test_validation_score_weights_practice_and_delayed_equally` — **PASS**
9. `EvaluationTests.test_oracle_probability_respects_format_and_strength` — **PASS**
10. `RecurrentTests.test_batched_replay_matches_incremental_including_repeated_queries` — **PASS**
11. `RecurrentTests.test_monotonic_and_kt_decay_satisfy_all_behavioral_sanity_checks` — **PASS**
12. `RecurrentTests.test_decay_is_identity_at_zero_and_converges_toward_prior` — **PASS**
13. `RecurrentTests.test_decay_and_recurrent_parameters_receive_gradients` — **PASS**
