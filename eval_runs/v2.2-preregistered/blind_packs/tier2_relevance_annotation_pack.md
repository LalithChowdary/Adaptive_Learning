# Tier 2 Blind Relevance Annotation Pack

- **Blinding Protocol:** Candidate documents are pooled across executed retrieval arms and deterministically shuffled per `retrieval_id` (`seed=20261201`). Arm names and retrieval ranks are strictly hidden.
- **Scale (`0 / 1 / 2`):** `0 = Irrelevant`, `1 = Partially relevant / helpful prerequisite`, `2 = Directly relevant`.
- **Pending Arms:** `dense`, `bm25_dense` (to be merged into candidate pools once dense weights are installed and selected on Dev).
- **Total Candidate Pairs Currently Pooled (Offline Arms):** 1090 across `66` Dev + `132` Held-Out Test queries.

| Annotation ID | Split | Retrieval ID | Context | Skill | Candidate Doc ID | Source | Section | Grade (0/1/2) |
|---|---|---|---|---|---|---|---|---|
| T2A-0001 | dev | `dev_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0002 | dev | `dev_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0003 | dev | `dev_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0004 | dev | `dev_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0005 | dev | `dev_chat_002__chat` | chat | range_boundaries | `ep.dev_learner_mastered.range_boundaries.da73d12e7b0d` | episodic | Learner History · doubt | |
| T2A-0006 | dev | `dev_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0007 | dev | `dev_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0008 | dev | `dev_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0009 | dev | `dev_chat_003__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.75f4e4325fb9` | episodic | Learner History · mistake | |
| T2A-0010 | dev | `dev_chat_003__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.b33bccf77b25` | episodic | Learner History · mistake | |
| T2A-0011 | dev | `dev_chat_003__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.be48dd8bd91e` | episodic | Learner History · reveal | |
| T2A-0012 | dev | `dev_chat_003__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.b134b29a4162` | episodic | Learner History · assisted_attempt | |
| T2A-0013 | dev | `dev_chat_004__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0014 | dev | `dev_chat_004__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.b134b29a4162` | episodic | Learner History · assisted_attempt | |
| T2A-0015 | dev | `dev_chat_004__chat` | chat | range_boundaries | `ep.dev_learner_lacking.range_boundaries.75f4e4325fb9` | episodic | Learner History · mistake | |
| T2A-0016 | dev | `dev_chat_004__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0017 | dev | `dev_chat_005__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0018 | dev | `dev_chat_005__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0019 | dev | `dev_chat_005__chat` | chat | range_boundaries | `ep.dev_learner_fragile.range_boundaries.39e8d1725062` | episodic | Learner History · doubt | |
| T2A-0020 | dev | `dev_chat_005__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0021 | dev | `dev_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0022 | dev | `dev_chat_006__chat` | chat | range_boundaries | `ep.dev_learner_fragile.range_boundaries.39e8d1725062` | episodic | Learner History · doubt | |
| T2A-0023 | dev | `dev_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0024 | dev | `dev_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0025 | dev | `dev_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0026 | dev | `dev_chat_007__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0027 | dev | `dev_chat_007__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0028 | dev | `dev_chat_007__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0029 | dev | `dev_chat_007__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0030 | dev | `dev_chat_008__chat` | chat | loop_tracing | `ep.dev_learner_mastered.loop_tracing.ddd777c8930e` | episodic | Learner History · doubt | |
| T2A-0031 | dev | `dev_chat_008__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0032 | dev | `dev_chat_008__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0033 | dev | `dev_chat_008__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0034 | dev | `dev_chat_009__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.a08b4e113bf9` | episodic | Learner History · mistake | |
| T2A-0035 | dev | `dev_chat_009__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.c810e6dd29aa` | episodic | Learner History · mistake | |
| T2A-0036 | dev | `dev_chat_009__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.eeb9a04bcaa1` | episodic | Learner History · reveal | |
| T2A-0037 | dev | `dev_chat_009__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.ef4ca2894125` | episodic | Learner History · assisted_attempt | |
| T2A-0038 | dev | `dev_chat_010__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.c810e6dd29aa` | episodic | Learner History · mistake | |
| T2A-0039 | dev | `dev_chat_010__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.ef4ca2894125` | episodic | Learner History · assisted_attempt | |
| T2A-0040 | dev | `dev_chat_010__chat` | chat | loop_tracing | `ep.dev_learner_lacking.loop_tracing.a08b4e113bf9` | episodic | Learner History · mistake | |
| T2A-0041 | dev | `dev_chat_010__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0042 | dev | `dev_chat_010__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0043 | dev | `dev_chat_011__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0044 | dev | `dev_chat_011__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0045 | dev | `dev_chat_011__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0046 | dev | `dev_chat_011__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0047 | dev | `dev_chat_011__chat` | chat | loop_tracing | `ep.dev_learner_fragile.loop_tracing.bea9082896a7` | episodic | Learner History · assisted_attempt | |
| T2A-0048 | dev | `dev_chat_012__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0049 | dev | `dev_chat_012__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0050 | dev | `dev_chat_012__chat` | chat | loop_tracing | `ep.dev_learner_fragile.loop_tracing.bea9082896a7` | episodic | Learner History · assisted_attempt | |
| T2A-0051 | dev | `dev_chat_012__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0052 | dev | `dev_chat_013__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.4.0` | curriculum | Prerequisite Connections | |
| T2A-0053 | dev | `dev_chat_013__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0054 | dev | `dev_chat_013__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0055 | dev | `dev_chat_013__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0056 | dev | `dev_chat_014__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0057 | dev | `dev_chat_014__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0058 | dev | `dev_chat_014__chat` | chat | accumulator_initialization | `ep.dev_learner_mastered.accumulator_initialization.613f21eae8dc` | episodic | Learner History · doubt | |
| T2A-0059 | dev | `dev_chat_014__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0060 | dev | `dev_chat_015__chat` | chat | accumulator_initialization | `ep.dev_learner_lacking.accumulator_initialization.449aa31cc291` | episodic | Learner History · mistake | |
| T2A-0061 | dev | `dev_chat_015__chat` | chat | accumulator_initialization | `ep.dev_learner_lacking.accumulator_initialization.5e86f77b3eac` | episodic | Learner History · mistake | |
| T2A-0062 | dev | `dev_chat_015__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0063 | dev | `dev_chat_015__chat` | chat | accumulator_initialization | `ep.dev_learner_lacking.accumulator_initialization.c2b4d09a9dbe` | episodic | Learner History · assisted_attempt | |
| T2A-0064 | dev | `dev_chat_016__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0065 | dev | `dev_chat_016__chat` | chat | accumulator_initialization | `ep.dev_learner_lacking.accumulator_initialization.449aa31cc291` | episodic | Learner History · mistake | |
| T2A-0066 | dev | `dev_chat_016__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0067 | dev | `dev_chat_016__chat` | chat | accumulator_initialization | `ep.dev_learner_lacking.accumulator_initialization.cbb6c7cc4867` | episodic | Learner History · reveal | |
| T2A-0068 | dev | `dev_chat_017__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0069 | dev | `dev_chat_017__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0070 | dev | `dev_chat_017__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0071 | dev | `dev_chat_017__chat` | chat | accumulator_initialization | `ep.dev_learner_fragile.accumulator_initialization.53f55a02a452` | episodic | Learner History · assisted_attempt | |
| T2A-0072 | dev | `dev_chat_018__chat` | chat | accumulator_initialization | `ep.dev_learner_fragile.accumulator_initialization.8aa337fea023` | episodic | Learner History · reveal | |
| T2A-0073 | dev | `dev_chat_018__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0074 | dev | `dev_chat_018__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0075 | dev | `dev_chat_018__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0076 | dev | `dev_chat_018__chat` | chat | accumulator_initialization | `ep.dev_learner_fragile.accumulator_initialization.53f55a02a452` | episodic | Learner History · assisted_attempt | |
| T2A-0077 | dev | `dev_chat_019__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0078 | dev | `dev_chat_019__chat` | chat | conditional_evaluation | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0079 | dev | `dev_chat_019__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0080 | dev | `dev_chat_019__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0081 | dev | `dev_chat_020__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0082 | dev | `dev_chat_020__chat` | chat | conditional_evaluation | `ep.dev_learner_mastered.conditional_evaluation.4c93857fa726` | episodic | Learner History · doubt | |
| T2A-0083 | dev | `dev_chat_020__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0084 | dev | `dev_chat_020__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0085 | dev | `dev_chat_020__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0086 | dev | `dev_chat_021__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0087 | dev | `dev_chat_021__chat` | chat | conditional_evaluation | `ep.dev_learner_lacking.conditional_evaluation.465f20fe1fa8` | episodic | Learner History · reveal | |
| T2A-0088 | dev | `dev_chat_021__chat` | chat | conditional_evaluation | `ep.dev_learner_lacking.conditional_evaluation.ad4a01ef5913` | episodic | Learner History · mistake | |
| T2A-0089 | dev | `dev_chat_021__chat` | chat | conditional_evaluation | `ep.dev_learner_lacking.conditional_evaluation.792c8129082d` | episodic | Learner History · assisted_attempt | |
| T2A-0090 | dev | `dev_chat_022__chat` | chat | conditional_evaluation | `ep.dev_learner_lacking.conditional_evaluation.792c8129082d` | episodic | Learner History · assisted_attempt | |
| T2A-0091 | dev | `dev_chat_022__chat` | chat | conditional_evaluation | `ep.dev_learner_lacking.conditional_evaluation.ad4a01ef5913` | episodic | Learner History · mistake | |
| T2A-0092 | dev | `dev_chat_022__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0093 | dev | `dev_chat_022__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0094 | dev | `dev_chat_023__chat` | chat | conditional_evaluation | `ep.dev_learner_fragile.conditional_evaluation.c040c5b2c9e1` | episodic | Learner History · assisted_attempt | |
| T2A-0095 | dev | `dev_chat_023__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0096 | dev | `dev_chat_023__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0097 | dev | `dev_chat_023__chat` | chat | conditional_evaluation | `ep.dev_learner_fragile.conditional_evaluation.2d851f52efb6` | episodic | Learner History · doubt | |
| T2A-0098 | dev | `dev_chat_023__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0099 | dev | `dev_chat_024__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0100 | dev | `dev_chat_024__chat` | chat | conditional_evaluation | `ep.dev_learner_fragile.conditional_evaluation.b4573aee5a85` | episodic | Learner History · reveal | |
| T2A-0101 | dev | `dev_chat_024__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0102 | dev | `dev_chat_024__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0103 | dev | `dev_chat_025__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0104 | dev | `dev_chat_025__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0105 | dev | `dev_chat_025__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0106 | dev | `dev_chat_025__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0107 | dev | `dev_chat_026__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0108 | dev | `dev_chat_026__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0109 | dev | `dev_chat_026__chat` | chat | list_indexing | `ep.dev_learner_mastered.list_indexing.56723ee33020` | episodic | Learner History · doubt | |
| T2A-0110 | dev | `dev_chat_026__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0111 | dev | `dev_chat_027__chat` | chat | list_indexing | `ep.dev_learner_lacking.list_indexing.ce6188d46715` | episodic | Learner History · reveal | |
| T2A-0112 | dev | `dev_chat_027__chat` | chat | list_indexing | `ep.dev_learner_lacking.list_indexing.0c8edcca8593` | episodic | Learner History · assisted_attempt | |
| T2A-0113 | dev | `dev_chat_027__chat` | chat | list_indexing | `ep.dev_learner_lacking.list_indexing.b19ea99c412c` | episodic | Learner History · mistake | |
| T2A-0114 | dev | `dev_chat_027__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0115 | dev | `dev_chat_028__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0116 | dev | `dev_chat_028__chat` | chat | list_indexing | `ep.dev_learner_lacking.list_indexing.0c8edcca8593` | episodic | Learner History · assisted_attempt | |
| T2A-0117 | dev | `dev_chat_028__chat` | chat | list_indexing | `ep.dev_learner_lacking.list_indexing.5a40a2c0394e` | episodic | Learner History · mistake | |
| T2A-0118 | dev | `dev_chat_028__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0119 | dev | `dev_chat_029__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0120 | dev | `dev_chat_029__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0121 | dev | `dev_chat_029__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0122 | dev | `dev_chat_029__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0123 | dev | `dev_chat_030__chat` | chat | list_indexing | `ep.dev_learner_fragile.list_indexing.233cf9bd6fa1` | episodic | Learner History · assisted_attempt | |
| T2A-0124 | dev | `dev_chat_030__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0125 | dev | `dev_chat_030__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0126 | dev | `dev_chat_030__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0127 | dev | `dev_chat_030__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0128 | dev | `dev_chat_031__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0129 | dev | `dev_chat_031__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0130 | dev | `dev_chat_031__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0131 | dev | `dev_chat_031__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0132 | dev | `dev_chat_032__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0133 | dev | `dev_chat_032__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0134 | dev | `dev_chat_032__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0135 | dev | `dev_chat_032__chat` | chat | function_returns | `ep.dev_learner_mastered.function_returns.620bfb78feff` | episodic | Learner History · doubt | |
| T2A-0136 | dev | `dev_chat_033__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0137 | dev | `dev_chat_033__chat` | chat | function_returns | `ep.dev_learner_lacking.function_returns.8fe1d658fbd6` | episodic | Learner History · mistake | |
| T2A-0138 | dev | `dev_chat_033__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0139 | dev | `dev_chat_033__chat` | chat | function_returns | `ep.dev_learner_lacking.function_returns.ddc463457314` | episodic | Learner History · reveal | |
| T2A-0140 | dev | `dev_chat_034__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0141 | dev | `dev_chat_034__chat` | chat | function_returns | `ep.dev_learner_lacking.function_returns.8fe1d658fbd6` | episodic | Learner History · mistake | |
| T2A-0142 | dev | `dev_chat_034__chat` | chat | function_returns | `ep.dev_learner_lacking.function_returns.ddc463457314` | episodic | Learner History · reveal | |
| T2A-0143 | dev | `dev_chat_034__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0144 | dev | `dev_chat_035__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0145 | dev | `dev_chat_035__chat` | chat | function_returns | `ep.dev_learner_fragile.function_returns.3f44378ff271` | episodic | Learner History · assisted_attempt | |
| T2A-0146 | dev | `dev_chat_035__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0147 | dev | `dev_chat_035__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0148 | dev | `dev_chat_035__chat` | chat | function_returns | `ep.dev_learner_fragile.function_returns.7f04546ef67e` | episodic | Learner History · reveal | |
| T2A-0149 | dev | `dev_chat_036__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0150 | dev | `dev_chat_036__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0151 | dev | `dev_chat_036__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0152 | dev | `dev_chat_036__chat` | chat | function_returns | `ep.dev_learner_fragile.function_returns.44f2b2731d01` | episodic | Learner History · doubt | |
| T2A-0153 | dev | `dev_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.single_choice.06` | v22_exemplar | easy · single_choice | |
| T2A-0154 | dev | `dev_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.single_choice.04` | v22_exemplar | easy · single_choice | |
| T2A-0155 | dev | `dev_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.single_choice.10` | v22_exemplar | easy · single_choice | |
| T2A-0156 | dev | `dev_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.single_choice.02` | v22_exemplar | easy · single_choice | |
| T2A-0157 | dev | `dev_qgen_001__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.c02817b94c89` | episodic | Learner History · assisted_attempt | |
| T2A-0158 | dev | `dev_qgen_001__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.750e434f96f7` | episodic | Learner History · mistake | |
| T2A-0159 | dev | `dev_qgen_001__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.90ecb5b8fc1d` | episodic | Learner History · mistake | |
| T2A-0160 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.03` | v22_exemplar | medium · multiple_select | |
| T2A-0161 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-0162 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.02` | v22_exemplar | medium · multiple_select | |
| T2A-0163 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-0164 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.04` | v22_exemplar | medium · multiple_select | |
| T2A-0165 | dev | `dev_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-0166 | dev | `dev_qgen_002__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0167 | dev | `dev_qgen_002__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0168 | dev | `dev_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.output_prediction.06` | v22_exemplar | hard · output_prediction | |
| T2A-0169 | dev | `dev_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.output_prediction.03` | v22_exemplar | hard · output_prediction | |
| T2A-0170 | dev | `dev_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.output_prediction.09` | v22_exemplar | hard · output_prediction | |
| T2A-0171 | dev | `dev_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.output_prediction.18` | v22_exemplar | hard · output_prediction | |
| T2A-0172 | dev | `dev_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.output_prediction.12` | v22_exemplar | hard · output_prediction | |
| T2A-0173 | dev | `dev_qgen_003__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.750e434f96f7` | episodic | Learner History · mistake | |
| T2A-0174 | dev | `dev_qgen_003__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.c02817b94c89` | episodic | Learner History · assisted_attempt | |
| T2A-0175 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.04` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0176 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.09` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0177 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0178 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.03` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0179 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.13` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0180 | dev | `dev_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.fill_in_the_blank.10` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0181 | dev | `dev_qgen_004__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0182 | dev | `dev_qgen_004__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0183 | dev | `dev_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.code_completion.03` | v22_exemplar | medium · code_completion | |
| T2A-0184 | dev | `dev_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.code_completion.04` | v22_exemplar | medium · code_completion | |
| T2A-0185 | dev | `dev_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.code_completion.02` | v22_exemplar | medium · code_completion | |
| T2A-0186 | dev | `dev_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-0187 | dev | `dev_qgen_005__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.c02817b94c89` | episodic | Learner History · assisted_attempt | |
| T2A-0188 | dev | `dev_qgen_005__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.dev_qgen_profile_a.range_boundaries.750e434f96f7` | episodic | Learner History · mistake | |
| T2A-0189 | dev | `dev_qgen_006__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.single_choice.06` | v22_exemplar | medium · single_choice | |
| T2A-0190 | dev | `dev_qgen_006__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.single_choice.12` | v22_exemplar | medium · single_choice | |
| T2A-0191 | dev | `dev_qgen_006__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.single_choice.02` | v22_exemplar | medium · single_choice | |
| T2A-0192 | dev | `dev_qgen_006__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.single_choice.04` | v22_exemplar | medium · single_choice | |
| T2A-0193 | dev | `dev_qgen_006__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0194 | dev | `dev_qgen_006__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0195 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-0196 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.04` | v22_exemplar | hard · multiple_select | |
| T2A-0197 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.03` | v22_exemplar | hard · multiple_select | |
| T2A-0198 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-0199 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-0200 | dev | `dev_qgen_007__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.multiple_select.02` | v22_exemplar | hard · multiple_select | |
| T2A-0201 | dev | `dev_qgen_007__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.dev_qgen_profile_a.loop_tracing.341e2fd3bddd` | episodic | Learner History · assisted_attempt | |
| T2A-0202 | dev | `dev_qgen_007__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.dev_qgen_profile_a.loop_tracing.0f19e88e5df1` | episodic | Learner History · mistake | |
| T2A-0203 | dev | `dev_qgen_008__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.output_prediction.10` | v22_exemplar | easy · output_prediction | |
| T2A-0204 | dev | `dev_qgen_008__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.output_prediction.02` | v22_exemplar | easy · output_prediction | |
| T2A-0205 | dev | `dev_qgen_008__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.output_prediction.11` | v22_exemplar | easy · output_prediction | |
| T2A-0206 | dev | `dev_qgen_008__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.output_prediction.05` | v22_exemplar | easy · output_prediction | |
| T2A-0207 | dev | `dev_qgen_008__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.output_prediction.04` | v22_exemplar | easy · output_prediction | |
| T2A-0208 | dev | `dev_qgen_008__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0209 | dev | `dev_qgen_008__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0210 | dev | `dev_qgen_009__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.fill_in_the_blank.03` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0211 | dev | `dev_qgen_009__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.fill_in_the_blank.21` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0212 | dev | `dev_qgen_009__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.fill_in_the_blank.09` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0213 | dev | `dev_qgen_009__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.fill_in_the_blank.15` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0214 | dev | `dev_qgen_009__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.fill_in_the_blank.06` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0215 | dev | `dev_qgen_009__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.dev_qgen_profile_a.loop_tracing.0f19e88e5df1` | episodic | Learner History · mistake | |
| T2A-0216 | dev | `dev_qgen_009__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.dev_qgen_profile_a.loop_tracing.341e2fd3bddd` | episodic | Learner History · assisted_attempt | |
| T2A-0217 | dev | `dev_qgen_010__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.code_completion.03` | v22_exemplar | hard · code_completion | |
| T2A-0218 | dev | `dev_qgen_010__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.code_completion.04` | v22_exemplar | hard · code_completion | |
| T2A-0219 | dev | `dev_qgen_010__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.code_completion.02` | v22_exemplar | hard · code_completion | |
| T2A-0220 | dev | `dev_qgen_010__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0221 | dev | `dev_qgen_010__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0222 | dev | `dev_qgen_011__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.single_choice.10` | v22_exemplar | hard · single_choice | |
| T2A-0223 | dev | `dev_qgen_011__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.single_choice.06` | v22_exemplar | hard · single_choice | |
| T2A-0224 | dev | `dev_qgen_011__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.single_choice.02` | v22_exemplar | hard · single_choice | |
| T2A-0225 | dev | `dev_qgen_011__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.single_choice.04` | v22_exemplar | hard · single_choice | |
| T2A-0226 | dev | `dev_qgen_011__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.67bab1dbc7ce` | episodic | Learner History · assisted_attempt | |
| T2A-0227 | dev | `dev_qgen_011__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.74e10f9ffaf3` | episodic | Learner History · mistake | |
| T2A-0228 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.04` | v22_exemplar | easy · multiple_select | |
| T2A-0229 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0230 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.02` | v22_exemplar | easy · multiple_select | |
| T2A-0231 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0232 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.03` | v22_exemplar | easy · multiple_select | |
| T2A-0233 | dev | `dev_qgen_012__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0234 | dev | `dev_qgen_012__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0235 | dev | `dev_qgen_012__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0236 | dev | `dev_qgen_013__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.output_prediction.09` | v22_exemplar | medium · output_prediction | |
| T2A-0237 | dev | `dev_qgen_013__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.output_prediction.03` | v22_exemplar | medium · output_prediction | |
| T2A-0238 | dev | `dev_qgen_013__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.output_prediction.06` | v22_exemplar | medium · output_prediction | |
| T2A-0239 | dev | `dev_qgen_013__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.output_prediction.15` | v22_exemplar | medium · output_prediction | |
| T2A-0240 | dev | `dev_qgen_013__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0241 | dev | `dev_qgen_013__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.67bab1dbc7ce` | episodic | Learner History · assisted_attempt | |
| T2A-0242 | dev | `dev_qgen_013__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.74e10f9ffaf3` | episodic | Learner History · mistake | |
| T2A-0243 | dev | `dev_qgen_014__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.fill_in_the_blank.15` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0244 | dev | `dev_qgen_014__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.fill_in_the_blank.03` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0245 | dev | `dev_qgen_014__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.fill_in_the_blank.09` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0246 | dev | `dev_qgen_014__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.fill_in_the_blank.21` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0247 | dev | `dev_qgen_014__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.fill_in_the_blank.06` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0248 | dev | `dev_qgen_014__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0249 | dev | `dev_qgen_014__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0250 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.02` | v22_exemplar | easy · code_completion | |
| T2A-0251 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-0252 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-0253 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-0254 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.03` | v22_exemplar | easy · code_completion | |
| T2A-0255 | dev | `dev_qgen_015__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.code_completion.04` | v22_exemplar | easy · code_completion | |
| T2A-0256 | dev | `dev_qgen_015__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.74e10f9ffaf3` | episodic | Learner History · mistake | |
| T2A-0257 | dev | `dev_qgen_015__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.dev_qgen_profile_a.accumulator_initialization.67bab1dbc7ce` | episodic | Learner History · assisted_attempt | |
| T2A-0258 | dev | `dev_qgen_016__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.single_choice.03` | v22_exemplar | easy · single_choice | |
| T2A-0259 | dev | `dev_qgen_016__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.single_choice.05` | v22_exemplar | easy · single_choice | |
| T2A-0260 | dev | `dev_qgen_016__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.single_choice.07` | v22_exemplar | easy · single_choice | |
| T2A-0261 | dev | `dev_qgen_016__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0262 | dev | `dev_qgen_016__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0263 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.03` | v22_exemplar | medium · multiple_select | |
| T2A-0264 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.04` | v22_exemplar | medium · multiple_select | |
| T2A-0265 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-0266 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-0267 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.02` | v22_exemplar | medium · multiple_select | |
| T2A-0268 | dev | `dev_qgen_017__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-0269 | dev | `dev_qgen_017__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0270 | dev | `dev_qgen_017__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.dev_qgen_profile_a.conditional_evaluation.0747711b742f` | episodic | Learner History · assisted_attempt | |
| T2A-0271 | dev | `dev_qgen_017__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.dev_qgen_profile_a.conditional_evaluation.7fddd9d22ccd` | episodic | Learner History · reveal | |
| T2A-0272 | dev | `dev_qgen_018__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.output_prediction.10` | v22_exemplar | hard · output_prediction | |
| T2A-0273 | dev | `dev_qgen_018__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.output_prediction.02` | v22_exemplar | hard · output_prediction | |
| T2A-0274 | dev | `dev_qgen_018__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.output_prediction.04` | v22_exemplar | hard · output_prediction | |
| T2A-0275 | dev | `dev_qgen_018__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.output_prediction.05` | v22_exemplar | hard · output_prediction | |
| T2A-0276 | dev | `dev_qgen_018__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0277 | dev | `dev_qgen_018__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0278 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.07` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0279 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.22` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0280 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.03` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0281 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0282 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.16` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0283 | dev | `dev_qgen_019__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.fill_in_the_blank.09` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0284 | dev | `dev_qgen_019__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.dev_qgen_profile_a.conditional_evaluation.0747711b742f` | episodic | Learner History · assisted_attempt | |
| T2A-0285 | dev | `dev_qgen_019__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0286 | dev | `dev_qgen_019__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.dev_qgen_profile_a.conditional_evaluation.7fddd9d22ccd` | episodic | Learner History · reveal | |
| T2A-0287 | dev | `dev_qgen_020__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.code_completion.02` | v22_exemplar | medium · code_completion | |
| T2A-0288 | dev | `dev_qgen_020__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.code_completion.03` | v22_exemplar | medium · code_completion | |
| T2A-0289 | dev | `dev_qgen_020__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.code_completion.04` | v22_exemplar | medium · code_completion | |
| T2A-0290 | dev | `dev_qgen_020__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-0291 | dev | `dev_qgen_020__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0292 | dev | `dev_qgen_020__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0293 | dev | `dev_qgen_021__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.single_choice.18` | v22_exemplar | medium · single_choice | |
| T2A-0294 | dev | `dev_qgen_021__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.single_choice.12` | v22_exemplar | medium · single_choice | |
| T2A-0295 | dev | `dev_qgen_021__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.single_choice.06` | v22_exemplar | medium · single_choice | |
| T2A-0296 | dev | `dev_qgen_021__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.ff1954783db1` | episodic | Learner History · assisted_attempt | |
| T2A-0297 | dev | `dev_qgen_021__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.f3ad41c0cede` | episodic | Learner History · mistake | |
| T2A-0298 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.03` | v22_exemplar | hard · multiple_select | |
| T2A-0299 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-0300 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.02` | v22_exemplar | hard · multiple_select | |
| T2A-0301 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-0302 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.04` | v22_exemplar | hard · multiple_select | |
| T2A-0303 | dev | `dev_qgen_022__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-0304 | dev | `dev_qgen_022__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0305 | dev | `dev_qgen_022__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0306 | dev | `dev_qgen_023__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.output_prediction.04` | v22_exemplar | easy · output_prediction | |
| T2A-0307 | dev | `dev_qgen_023__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.output_prediction.05` | v22_exemplar | easy · output_prediction | |
| T2A-0308 | dev | `dev_qgen_023__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.output_prediction.02` | v22_exemplar | easy · output_prediction | |
| T2A-0309 | dev | `dev_qgen_023__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.output_prediction.10` | v22_exemplar | easy · output_prediction | |
| T2A-0310 | dev | `dev_qgen_023__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.ff1954783db1` | episodic | Learner History · assisted_attempt | |
| T2A-0311 | dev | `dev_qgen_023__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.f3ad41c0cede` | episodic | Learner History · mistake | |
| T2A-0312 | dev | `dev_qgen_024__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.fill_in_the_blank.13` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0313 | dev | `dev_qgen_024__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.fill_in_the_blank.19` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0314 | dev | `dev_qgen_024__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.fill_in_the_blank.07` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0315 | dev | `dev_qgen_024__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0316 | dev | `dev_qgen_024__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0317 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.04` | v22_exemplar | hard · code_completion | |
| T2A-0318 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.02` | v22_exemplar | hard · code_completion | |
| T2A-0319 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.11` | v22_exemplar | hard · code_completion | |
| T2A-0320 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.12` | v22_exemplar | hard · code_completion | |
| T2A-0321 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.03` | v22_exemplar | hard · code_completion | |
| T2A-0322 | dev | `dev_qgen_025__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.code_completion.10` | v22_exemplar | hard · code_completion | |
| T2A-0323 | dev | `dev_qgen_025__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.f3ad41c0cede` | episodic | Learner History · mistake | |
| T2A-0324 | dev | `dev_qgen_025__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.dev_qgen_profile_a.list_indexing.ff1954783db1` | episodic | Learner History · assisted_attempt | |
| T2A-0325 | dev | `dev_qgen_026__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.single_choice.07` | v22_exemplar | hard · single_choice | |
| T2A-0326 | dev | `dev_qgen_026__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.single_choice.03` | v22_exemplar | hard · single_choice | |
| T2A-0327 | dev | `dev_qgen_026__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.single_choice.05` | v22_exemplar | hard · single_choice | |
| T2A-0328 | dev | `dev_qgen_026__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.single_choice.11` | v22_exemplar | hard · single_choice | |
| T2A-0329 | dev | `dev_qgen_026__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0330 | dev | `dev_qgen_026__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0331 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.04` | v22_exemplar | easy · multiple_select | |
| T2A-0332 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.03` | v22_exemplar | easy · multiple_select | |
| T2A-0333 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0334 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.02` | v22_exemplar | easy · multiple_select | |
| T2A-0335 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0336 | dev | `dev_qgen_027__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0337 | dev | `dev_qgen_027__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.1cb0fcf65af3` | episodic | Learner History · reveal | |
| T2A-0338 | dev | `dev_qgen_027__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.1f13e814791e` | episodic | Learner History · mistake | |
| T2A-0339 | dev | `dev_qgen_027__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.85e72a7239b1` | episodic | Learner History · assisted_attempt | |
| T2A-0340 | dev | `dev_qgen_028__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.output_prediction.06` | v22_exemplar | medium · output_prediction | |
| T2A-0341 | dev | `dev_qgen_028__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.output_prediction.15` | v22_exemplar | medium · output_prediction | |
| T2A-0342 | dev | `dev_qgen_028__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.output_prediction.09` | v22_exemplar | medium · output_prediction | |
| T2A-0343 | dev | `dev_qgen_028__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.output_prediction.03` | v22_exemplar | medium · output_prediction | |
| T2A-0344 | dev | `dev_qgen_028__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0345 | dev | `dev_qgen_028__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0346 | dev | `dev_qgen_028__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0347 | dev | `dev_qgen_029__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.fill_in_the_blank.03` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0348 | dev | `dev_qgen_029__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.fill_in_the_blank.09` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0349 | dev | `dev_qgen_029__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.fill_in_the_blank.10` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0350 | dev | `dev_qgen_029__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.fill_in_the_blank.06` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0351 | dev | `dev_qgen_029__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.fill_in_the_blank.16` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0352 | dev | `dev_qgen_029__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.1cb0fcf65af3` | episodic | Learner History · reveal | |
| T2A-0353 | dev | `dev_qgen_029__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.85e72a7239b1` | episodic | Learner History · assisted_attempt | |
| T2A-0354 | dev | `dev_qgen_029__qgen_curriculum` | qgen_curriculum | function_returns | `ep.dev_qgen_profile_a.function_returns.1f13e814791e` | episodic | Learner History · mistake | |
| T2A-0355 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.04` | v22_exemplar | easy · code_completion | |
| T2A-0356 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-0357 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-0358 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.03` | v22_exemplar | easy · code_completion | |
| T2A-0359 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-0360 | dev | `dev_qgen_030__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.code_completion.02` | v22_exemplar | easy · code_completion | |
| T2A-0361 | dev | `dev_qgen_030__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0362 | dev | `dev_qgen_030__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0363 | test | `test_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0364 | test | `test_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0365 | test | `test_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0366 | test | `test_chat_001__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0367 | test | `test_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0368 | test | `test_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0369 | test | `test_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0370 | test | `test_chat_002__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0371 | test | `test_chat_003__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0372 | test | `test_chat_003__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0373 | test | `test_chat_003__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0374 | test | `test_chat_003__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0375 | test | `test_chat_004__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.8b890a10d98c` | episodic | Learner History · mistake | |
| T2A-0376 | test | `test_chat_004__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0377 | test | `test_chat_004__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.cc491be702e5` | episodic | Learner History · mistake | |
| T2A-0378 | test | `test_chat_004__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0379 | test | `test_chat_004__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.977b67158042` | episodic | Learner History · assisted_attempt | |
| T2A-0380 | test | `test_chat_005__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.977b67158042` | episodic | Learner History · assisted_attempt | |
| T2A-0381 | test | `test_chat_005__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0382 | test | `test_chat_005__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.cc491be702e5` | episodic | Learner History · mistake | |
| T2A-0383 | test | `test_chat_005__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.8b890a10d98c` | episodic | Learner History · mistake | |
| T2A-0384 | test | `test_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0385 | test | `test_chat_006__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.977b67158042` | episodic | Learner History · assisted_attempt | |
| T2A-0386 | test | `test_chat_006__chat` | chat | range_boundaries | `ep.test_learner_lacking.range_boundaries.cc491be702e5` | episodic | Learner History · mistake | |
| T2A-0387 | test | `test_chat_006__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0388 | test | `test_chat_007__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0389 | test | `test_chat_007__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0390 | test | `test_chat_007__chat` | chat | range_boundaries | `ep.test_learner_fragile.range_boundaries.bd134da40312` | episodic | Learner History · assisted_attempt | |
| T2A-0391 | test | `test_chat_007__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0392 | test | `test_chat_008__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0393 | test | `test_chat_008__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0394 | test | `test_chat_008__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0395 | test | `test_chat_008__chat` | chat | range_boundaries | `ep.test_learner_fragile.range_boundaries.bd134da40312` | episodic | Learner History · assisted_attempt | |
| T2A-0396 | test | `test_chat_009__chat` | chat | range_boundaries | `ep.test_learner_fragile.range_boundaries.bd134da40312` | episodic | Learner History · assisted_attempt | |
| T2A-0397 | test | `test_chat_009__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0398 | test | `test_chat_009__chat` | chat | range_boundaries | `ep.test_learner_fragile.range_boundaries.e8a50a2f36be` | episodic | Learner History · doubt | |
| T2A-0399 | test | `test_chat_009__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0400 | test | `test_chat_009__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0401 | test | `test_chat_010__chat` | chat | range_boundaries | `ep.test_learner_mastered.range_boundaries.c8d638bc00b3` | episodic | Learner History · doubt | |
| T2A-0402 | test | `test_chat_010__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0403 | test | `test_chat_010__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0404 | test | `test_chat_010__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0405 | test | `test_chat_011__chat` | chat | range_boundaries | `ep.test_learner_mastered.range_boundaries.c8d638bc00b3` | episodic | Learner History · doubt | |
| T2A-0406 | test | `test_chat_011__chat` | chat | range_boundaries | `ep.test_learner_mastered.range_boundaries.ec3dcbd4464f` | episodic | Learner History · doubt | |
| T2A-0407 | test | `test_chat_011__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0408 | test | `test_chat_011__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0409 | test | `test_chat_012__chat` | chat | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0410 | test | `test_chat_012__chat` | chat | range_boundaries | `cur.range_boundaries.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0411 | test | `test_chat_012__chat` | chat | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0412 | test | `test_chat_012__chat` | chat | range_boundaries | `cur.range_boundaries.4.0` | curriculum | Prerequisite Connections | |
| T2A-0413 | test | `test_chat_013__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0414 | test | `test_chat_013__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0415 | test | `test_chat_013__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0416 | test | `test_chat_013__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0417 | test | `test_chat_014__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0418 | test | `test_chat_014__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0419 | test | `test_chat_014__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0420 | test | `test_chat_014__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0421 | test | `test_chat_015__chat` | chat | loop_tracing | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0422 | test | `test_chat_015__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0423 | test | `test_chat_015__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0424 | test | `test_chat_015__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0425 | test | `test_chat_016__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.095aa771187f` | episodic | Learner History · assisted_attempt | |
| T2A-0426 | test | `test_chat_016__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0427 | test | `test_chat_016__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.58e410df3e82` | episodic | Learner History · mistake | |
| T2A-0428 | test | `test_chat_016__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.cdd39009a14c` | episodic | Learner History · reveal | |
| T2A-0429 | test | `test_chat_017__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.dc83f473d608` | episodic | Learner History · mistake | |
| T2A-0430 | test | `test_chat_017__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0431 | test | `test_chat_017__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.095aa771187f` | episodic | Learner History · assisted_attempt | |
| T2A-0432 | test | `test_chat_017__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0433 | test | `test_chat_017__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.58e410df3e82` | episodic | Learner History · mistake | |
| T2A-0434 | test | `test_chat_018__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0435 | test | `test_chat_018__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.58e410df3e82` | episodic | Learner History · mistake | |
| T2A-0436 | test | `test_chat_018__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.cdd39009a14c` | episodic | Learner History · reveal | |
| T2A-0437 | test | `test_chat_018__chat` | chat | loop_tracing | `ep.test_learner_lacking.loop_tracing.095aa771187f` | episodic | Learner History · assisted_attempt | |
| T2A-0438 | test | `test_chat_019__chat` | chat | loop_tracing | `ep.test_learner_fragile.loop_tracing.c59f969bd923` | episodic | Learner History · reveal | |
| T2A-0439 | test | `test_chat_019__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0440 | test | `test_chat_019__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0441 | test | `test_chat_019__chat` | chat | loop_tracing | `ep.test_learner_fragile.loop_tracing.c4d985c86b6f` | episodic | Learner History · doubt | |
| T2A-0442 | test | `test_chat_019__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0443 | test | `test_chat_020__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0444 | test | `test_chat_020__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0445 | test | `test_chat_020__chat` | chat | loop_tracing | `ep.test_learner_fragile.loop_tracing.90a5d75db89e` | episodic | Learner History · assisted_attempt | |
| T2A-0446 | test | `test_chat_020__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0447 | test | `test_chat_021__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0448 | test | `test_chat_021__chat` | chat | loop_tracing | `ep.test_learner_fragile.loop_tracing.90a5d75db89e` | episodic | Learner History · assisted_attempt | |
| T2A-0449 | test | `test_chat_021__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0450 | test | `test_chat_021__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0451 | test | `test_chat_022__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0452 | test | `test_chat_022__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0453 | test | `test_chat_022__chat` | chat | loop_tracing | `ep.test_learner_mastered.loop_tracing.b120cfc9cbca` | episodic | Learner History · doubt | |
| T2A-0454 | test | `test_chat_022__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0455 | test | `test_chat_023__chat` | chat | loop_tracing | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0456 | test | `test_chat_023__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0457 | test | `test_chat_023__chat` | chat | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0458 | test | `test_chat_023__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0459 | test | `test_chat_024__chat` | chat | loop_tracing | `cur.loop_tracing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0460 | test | `test_chat_024__chat` | chat | loop_tracing | `ep.test_learner_mastered.loop_tracing.a40520abb2d9` | episodic | Learner History · mistake | |
| T2A-0461 | test | `test_chat_024__chat` | chat | loop_tracing | `ep.test_learner_mastered.loop_tracing.b120cfc9cbca` | episodic | Learner History · doubt | |
| T2A-0462 | test | `test_chat_024__chat` | chat | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0463 | test | `test_chat_025__chat` | chat | accumulator_initialization | `cur.loop_tracing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0464 | test | `test_chat_025__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.4.0` | curriculum | Prerequisite Connections | |
| T2A-0465 | test | `test_chat_025__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0466 | test | `test_chat_025__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0467 | test | `test_chat_026__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0468 | test | `test_chat_026__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0469 | test | `test_chat_026__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0470 | test | `test_chat_026__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.4.0` | curriculum | Prerequisite Connections | |
| T2A-0471 | test | `test_chat_027__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.4.0` | curriculum | Prerequisite Connections | |
| T2A-0472 | test | `test_chat_027__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0473 | test | `test_chat_027__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0474 | test | `test_chat_027__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0475 | test | `test_chat_028__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.9d0f523bb65f` | episodic | Learner History · mistake | |
| T2A-0476 | test | `test_chat_028__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0477 | test | `test_chat_028__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.46d35f280ff6` | episodic | Learner History · assisted_attempt | |
| T2A-0478 | test | `test_chat_028__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.013be14f5d6d` | episodic | Learner History · mistake | |
| T2A-0479 | test | `test_chat_028__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0480 | test | `test_chat_029__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.9d0f523bb65f` | episodic | Learner History · mistake | |
| T2A-0481 | test | `test_chat_029__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0482 | test | `test_chat_029__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.013be14f5d6d` | episodic | Learner History · mistake | |
| T2A-0483 | test | `test_chat_029__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0484 | test | `test_chat_030__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0485 | test | `test_chat_030__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0486 | test | `test_chat_030__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.9d0f523bb65f` | episodic | Learner History · mistake | |
| T2A-0487 | test | `test_chat_030__chat` | chat | accumulator_initialization | `ep.test_learner_lacking.accumulator_initialization.013be14f5d6d` | episodic | Learner History · mistake | |
| T2A-0488 | test | `test_chat_031__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0489 | test | `test_chat_031__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0490 | test | `test_chat_031__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0491 | test | `test_chat_031__chat` | chat | accumulator_initialization | `ep.test_learner_fragile.accumulator_initialization.1ffa63da2ac8` | episodic | Learner History · assisted_attempt | |
| T2A-0492 | test | `test_chat_032__chat` | chat | accumulator_initialization | `ep.test_learner_fragile.accumulator_initialization.e941a5ef8905` | episodic | Learner History · doubt | |
| T2A-0493 | test | `test_chat_032__chat` | chat | accumulator_initialization | `ep.test_learner_fragile.accumulator_initialization.1ffa63da2ac8` | episodic | Learner History · assisted_attempt | |
| T2A-0494 | test | `test_chat_032__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0495 | test | `test_chat_032__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0496 | test | `test_chat_032__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0497 | test | `test_chat_033__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0498 | test | `test_chat_033__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0499 | test | `test_chat_033__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0500 | test | `test_chat_033__chat` | chat | accumulator_initialization | `ep.test_learner_fragile.accumulator_initialization.1ffa63da2ac8` | episodic | Learner History · assisted_attempt | |
| T2A-0501 | test | `test_chat_033__chat` | chat | accumulator_initialization | `ep.test_learner_fragile.accumulator_initialization.e941a5ef8905` | episodic | Learner History · doubt | |
| T2A-0502 | test | `test_chat_034__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0503 | test | `test_chat_034__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0504 | test | `test_chat_034__chat` | chat | accumulator_initialization | `ep.test_learner_mastered.accumulator_initialization.ee110301ab2e` | episodic | Learner History · doubt | |
| T2A-0505 | test | `test_chat_034__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0506 | test | `test_chat_035__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0507 | test | `test_chat_035__chat` | chat | accumulator_initialization | `ep.test_learner_mastered.accumulator_initialization.ee110301ab2e` | episodic | Learner History · doubt | |
| T2A-0508 | test | `test_chat_035__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.4.0` | curriculum | Prerequisite Connections | |
| T2A-0509 | test | `test_chat_035__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0510 | test | `test_chat_035__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0511 | test | `test_chat_036__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0512 | test | `test_chat_036__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0513 | test | `test_chat_036__chat` | chat | accumulator_initialization | `ep.test_learner_mastered.accumulator_initialization.ee110301ab2e` | episodic | Learner History · doubt | |
| T2A-0514 | test | `test_chat_036__chat` | chat | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0515 | test | `test_chat_037__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0516 | test | `test_chat_037__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0517 | test | `test_chat_037__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0518 | test | `test_chat_037__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0519 | test | `test_chat_038__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0520 | test | `test_chat_038__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0521 | test | `test_chat_038__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0522 | test | `test_chat_038__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0523 | test | `test_chat_039__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0524 | test | `test_chat_039__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0525 | test | `test_chat_039__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0526 | test | `test_chat_039__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0527 | test | `test_chat_040__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.eb8ef129010a` | episodic | Learner History · assisted_attempt | |
| T2A-0528 | test | `test_chat_040__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.4744a6fade31` | episodic | Learner History · reveal | |
| T2A-0529 | test | `test_chat_040__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.839cbbe3599e` | episodic | Learner History · mistake | |
| T2A-0530 | test | `test_chat_040__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0531 | test | `test_chat_041__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.839cbbe3599e` | episodic | Learner History · mistake | |
| T2A-0532 | test | `test_chat_041__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.4744a6fade31` | episodic | Learner History · reveal | |
| T2A-0533 | test | `test_chat_041__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.eb8ef129010a` | episodic | Learner History · assisted_attempt | |
| T2A-0534 | test | `test_chat_041__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0535 | test | `test_chat_042__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.24bd245ecee5` | episodic | Learner History · mistake | |
| T2A-0536 | test | `test_chat_042__chat` | chat | conditional_evaluation | `ep.test_learner_lacking.conditional_evaluation.4744a6fade31` | episodic | Learner History · reveal | |
| T2A-0537 | test | `test_chat_042__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0538 | test | `test_chat_042__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0539 | test | `test_chat_043__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0540 | test | `test_chat_043__chat` | chat | conditional_evaluation | `ep.test_learner_fragile.conditional_evaluation.f33f7bd466ba` | episodic | Learner History · doubt | |
| T2A-0541 | test | `test_chat_043__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0542 | test | `test_chat_043__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0543 | test | `test_chat_044__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0544 | test | `test_chat_044__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0545 | test | `test_chat_044__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0546 | test | `test_chat_044__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0547 | test | `test_chat_045__chat` | chat | conditional_evaluation | `ep.test_learner_fragile.conditional_evaluation.53d1e6b35e0d` | episodic | Learner History · assisted_attempt | |
| T2A-0548 | test | `test_chat_045__chat` | chat | conditional_evaluation | `ep.test_learner_fragile.conditional_evaluation.1409cb8d9c38` | episodic | Learner History · reveal | |
| T2A-0549 | test | `test_chat_045__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0550 | test | `test_chat_045__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0551 | test | `test_chat_045__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0552 | test | `test_chat_046__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0553 | test | `test_chat_046__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0554 | test | `test_chat_046__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0555 | test | `test_chat_046__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0556 | test | `test_chat_047__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0557 | test | `test_chat_047__chat` | chat | conditional_evaluation | `ep.test_learner_mastered.conditional_evaluation.00d3712d7576` | episodic | Learner History · doubt | |
| T2A-0558 | test | `test_chat_047__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0559 | test | `test_chat_047__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0560 | test | `test_chat_048__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0561 | test | `test_chat_048__chat` | chat | conditional_evaluation | `ep.test_learner_mastered.conditional_evaluation.053fad068f69` | episodic | Learner History · doubt | |
| T2A-0562 | test | `test_chat_048__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0563 | test | `test_chat_048__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.4.0` | curriculum | Prerequisite Connections | |
| T2A-0564 | test | `test_chat_048__chat` | chat | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0565 | test | `test_chat_049__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0566 | test | `test_chat_049__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0567 | test | `test_chat_049__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0568 | test | `test_chat_049__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0569 | test | `test_chat_050__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0570 | test | `test_chat_050__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0571 | test | `test_chat_050__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0572 | test | `test_chat_050__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0573 | test | `test_chat_051__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0574 | test | `test_chat_051__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0575 | test | `test_chat_051__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0576 | test | `test_chat_051__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0577 | test | `test_chat_052__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.927622ec21f8` | episodic | Learner History · assisted_attempt | |
| T2A-0578 | test | `test_chat_052__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.e0015a9c683c` | episodic | Learner History · mistake | |
| T2A-0579 | test | `test_chat_052__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.750ffb55872b` | episodic | Learner History · reveal | |
| T2A-0580 | test | `test_chat_052__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0581 | test | `test_chat_053__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0582 | test | `test_chat_053__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0583 | test | `test_chat_053__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.3fe820de7449` | episodic | Learner History · mistake | |
| T2A-0584 | test | `test_chat_053__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.927622ec21f8` | episodic | Learner History · assisted_attempt | |
| T2A-0585 | test | `test_chat_054__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.3fe820de7449` | episodic | Learner History · mistake | |
| T2A-0586 | test | `test_chat_054__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.927622ec21f8` | episodic | Learner History · assisted_attempt | |
| T2A-0587 | test | `test_chat_054__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.750ffb55872b` | episodic | Learner History · reveal | |
| T2A-0588 | test | `test_chat_054__chat` | chat | list_indexing | `ep.test_learner_lacking.list_indexing.e0015a9c683c` | episodic | Learner History · mistake | |
| T2A-0589 | test | `test_chat_055__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0590 | test | `test_chat_055__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0591 | test | `test_chat_055__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.e19537ad5c51` | episodic | Learner History · doubt | |
| T2A-0592 | test | `test_chat_055__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0593 | test | `test_chat_056__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.e19537ad5c51` | episodic | Learner History · doubt | |
| T2A-0594 | test | `test_chat_056__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.e5253438ccc4` | episodic | Learner History · assisted_attempt | |
| T2A-0595 | test | `test_chat_056__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.a32c801c3d86` | episodic | Learner History · reveal | |
| T2A-0596 | test | `test_chat_056__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0597 | test | `test_chat_056__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0598 | test | `test_chat_057__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0599 | test | `test_chat_057__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.e19537ad5c51` | episodic | Learner History · doubt | |
| T2A-0600 | test | `test_chat_057__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0601 | test | `test_chat_057__chat` | chat | list_indexing | `ep.test_learner_fragile.list_indexing.e5253438ccc4` | episodic | Learner History · assisted_attempt | |
| T2A-0602 | test | `test_chat_057__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0603 | test | `test_chat_058__chat` | chat | list_indexing | `ep.test_learner_mastered.list_indexing.966ed3711a2b` | episodic | Learner History · doubt | |
| T2A-0604 | test | `test_chat_058__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0605 | test | `test_chat_058__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0606 | test | `test_chat_058__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0607 | test | `test_chat_059__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0608 | test | `test_chat_059__chat` | chat | list_indexing | `ep.test_learner_mastered.list_indexing.966ed3711a2b` | episodic | Learner History · doubt | |
| T2A-0609 | test | `test_chat_059__chat` | chat | list_indexing | `ep.test_learner_mastered.list_indexing.242fa07bee70` | episodic | Learner History · doubt | |
| T2A-0610 | test | `test_chat_059__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0611 | test | `test_chat_060__chat` | chat | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0612 | test | `test_chat_060__chat` | chat | list_indexing | `cur.list_indexing.4.0` | curriculum | Prerequisite Connections | |
| T2A-0613 | test | `test_chat_060__chat` | chat | list_indexing | `cur.list_indexing.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0614 | test | `test_chat_060__chat` | chat | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0615 | test | `test_chat_061__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0616 | test | `test_chat_061__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0617 | test | `test_chat_061__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0618 | test | `test_chat_061__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0619 | test | `test_chat_062__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0620 | test | `test_chat_062__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0621 | test | `test_chat_062__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0622 | test | `test_chat_062__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0623 | test | `test_chat_063__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0624 | test | `test_chat_063__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0625 | test | `test_chat_063__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0626 | test | `test_chat_063__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0627 | test | `test_chat_064__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0628 | test | `test_chat_064__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0629 | test | `test_chat_064__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.587e4ad623a6` | episodic | Learner History · reveal | |
| T2A-0630 | test | `test_chat_064__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.5d2bf187e7b0` | episodic | Learner History · mistake | |
| T2A-0631 | test | `test_chat_065__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.587e4ad623a6` | episodic | Learner History · reveal | |
| T2A-0632 | test | `test_chat_065__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.9092c030ae3c` | episodic | Learner History · mistake | |
| T2A-0633 | test | `test_chat_065__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0634 | test | `test_chat_065__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.5d2bf187e7b0` | episodic | Learner History · mistake | |
| T2A-0635 | test | `test_chat_066__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0636 | test | `test_chat_066__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.5d2bf187e7b0` | episodic | Learner History · mistake | |
| T2A-0637 | test | `test_chat_066__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.587e4ad623a6` | episodic | Learner History · reveal | |
| T2A-0638 | test | `test_chat_066__chat` | chat | function_returns | `ep.test_learner_lacking.function_returns.302bac312491` | episodic | Learner History · assisted_attempt | |
| T2A-0639 | test | `test_chat_067__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0640 | test | `test_chat_067__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0641 | test | `test_chat_067__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0642 | test | `test_chat_067__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0643 | test | `test_chat_068__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0644 | test | `test_chat_068__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0645 | test | `test_chat_068__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0646 | test | `test_chat_068__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0647 | test | `test_chat_068__chat` | chat | function_returns | `ep.test_learner_fragile.function_returns.23f0ec2b12bb` | episodic | Learner History · reveal | |
| T2A-0648 | test | `test_chat_069__chat` | chat | function_returns | `ep.test_learner_fragile.function_returns.5c0c19c29f18` | episodic | Learner History · doubt | |
| T2A-0649 | test | `test_chat_069__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0650 | test | `test_chat_069__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0651 | test | `test_chat_069__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0652 | test | `test_chat_069__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0653 | test | `test_chat_070__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0654 | test | `test_chat_070__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0655 | test | `test_chat_070__chat` | chat | function_returns | `ep.test_learner_mastered.function_returns.f412ca289c8c` | episodic | Learner History · doubt | |
| T2A-0656 | test | `test_chat_070__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0657 | test | `test_chat_070__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0658 | test | `test_chat_071__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0659 | test | `test_chat_071__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0660 | test | `test_chat_071__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0661 | test | `test_chat_071__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0662 | test | `test_chat_072__chat` | chat | function_returns | `cur.function_returns.1.0` | curriculum | Core Concepts and Mental Model | |
| T2A-0663 | test | `test_chat_072__chat` | chat | function_returns | `cur.function_returns.4.0` | curriculum | Prerequisite Connections | |
| T2A-0664 | test | `test_chat_072__chat` | chat | function_returns | `ep.test_learner_mastered.function_returns.f412ca289c8c` | episodic | Learner History · doubt | |
| T2A-0665 | test | `test_chat_072__chat` | chat | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0666 | test | `test_chat_072__chat` | chat | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0667 | test | `test_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.single_choice.10` | v22_exemplar | medium · single_choice | |
| T2A-0668 | test | `test_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.single_choice.06` | v22_exemplar | medium · single_choice | |
| T2A-0669 | test | `test_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.single_choice.08` | v22_exemplar | medium · single_choice | |
| T2A-0670 | test | `test_qgen_001__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.single_choice.00` | v22_exemplar | medium · single_choice | |
| T2A-0671 | test | `test_qgen_001__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.a39fe2fc2aae` | episodic | Learner History · mistake | |
| T2A-0672 | test | `test_qgen_001__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.cd7ac6bb2592` | episodic | Learner History · assisted_attempt | |
| T2A-0673 | test | `test_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.single_choice.06` | v22_exemplar | hard · single_choice | |
| T2A-0674 | test | `test_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.single_choice.00` | v22_exemplar | hard · single_choice | |
| T2A-0675 | test | `test_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.single_choice.08` | v22_exemplar | hard · single_choice | |
| T2A-0676 | test | `test_qgen_002__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.single_choice.10` | v22_exemplar | hard · single_choice | |
| T2A-0677 | test | `test_qgen_002__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0678 | test | `test_qgen_002__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0679 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.00` | v22_exemplar | easy · multiple_select | |
| T2A-0680 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0681 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.01` | v22_exemplar | easy · multiple_select | |
| T2A-0682 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.05` | v22_exemplar | easy · multiple_select | |
| T2A-0683 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0684 | test | `test_qgen_003__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0685 | test | `test_qgen_003__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.89873f43260a` | episodic | Learner History · mistake | |
| T2A-0686 | test | `test_qgen_003__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.cd7ac6bb2592` | episodic | Learner History · assisted_attempt | |
| T2A-0687 | test | `test_qgen_003__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.a39fe2fc2aae` | episodic | Learner History · mistake | |
| T2A-0688 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.01` | v22_exemplar | hard · multiple_select | |
| T2A-0689 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-0690 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-0691 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.05` | v22_exemplar | hard · multiple_select | |
| T2A-0692 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.00` | v22_exemplar | hard · multiple_select | |
| T2A-0693 | test | `test_qgen_004__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-0694 | test | `test_qgen_004__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0695 | test | `test_qgen_004__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0696 | test | `test_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.output_prediction.09` | v22_exemplar | easy · output_prediction | |
| T2A-0697 | test | `test_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.output_prediction.00` | v22_exemplar | easy · output_prediction | |
| T2A-0698 | test | `test_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.output_prediction.12` | v22_exemplar | easy · output_prediction | |
| T2A-0699 | test | `test_qgen_005__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.output_prediction.06` | v22_exemplar | easy · output_prediction | |
| T2A-0700 | test | `test_qgen_005__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.89873f43260a` | episodic | Learner History · mistake | |
| T2A-0701 | test | `test_qgen_005__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.a39fe2fc2aae` | episodic | Learner History · mistake | |
| T2A-0702 | test | `test_qgen_005__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.cd7ac6bb2592` | episodic | Learner History · assisted_attempt | |
| T2A-0703 | test | `test_qgen_006__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.output_prediction.06` | v22_exemplar | medium · output_prediction | |
| T2A-0704 | test | `test_qgen_006__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0705 | test | `test_qgen_006__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.output_prediction.09` | v22_exemplar | medium · output_prediction | |
| T2A-0706 | test | `test_qgen_006__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.output_prediction.15` | v22_exemplar | medium · output_prediction | |
| T2A-0707 | test | `test_qgen_006__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.output_prediction.00` | v22_exemplar | medium · output_prediction | |
| T2A-0708 | test | `test_qgen_006__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0709 | test | `test_qgen_006__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0710 | test | `test_qgen_007__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.fill_in_the_blank.09` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0711 | test | `test_qgen_007__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.fill_in_the_blank.00` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0712 | test | `test_qgen_007__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.fill_in_the_blank.06` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0713 | test | `test_qgen_007__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.fill_in_the_blank.08` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0714 | test | `test_qgen_007__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.medium.fill_in_the_blank.11` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0715 | test | `test_qgen_007__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.cd7ac6bb2592` | episodic | Learner History · assisted_attempt | |
| T2A-0716 | test | `test_qgen_007__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.a39fe2fc2aae` | episodic | Learner History · mistake | |
| T2A-0717 | test | `test_qgen_008__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.fill_in_the_blank.06` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0718 | test | `test_qgen_008__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.fill_in_the_blank.16` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0719 | test | `test_qgen_008__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.fill_in_the_blank.00` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0720 | test | `test_qgen_008__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.fill_in_the_blank.12` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0721 | test | `test_qgen_008__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.fill_in_the_blank.09` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0722 | test | `test_qgen_008__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0723 | test | `test_qgen_008__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0724 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.05` | v22_exemplar | easy · code_completion | |
| T2A-0725 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-0726 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-0727 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.01` | v22_exemplar | easy · code_completion | |
| T2A-0728 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-0729 | test | `test_qgen_009__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.easy.code_completion.00` | v22_exemplar | easy · code_completion | |
| T2A-0730 | test | `test_qgen_009__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.89873f43260a` | episodic | Learner History · mistake | |
| T2A-0731 | test | `test_qgen_009__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.cd7ac6bb2592` | episodic | Learner History · assisted_attempt | |
| T2A-0732 | test | `test_qgen_009__qgen_curriculum` | qgen_curriculum | range_boundaries | `ep.test_qgen_profile_a.range_boundaries.a39fe2fc2aae` | episodic | Learner History · mistake | |
| T2A-0733 | test | `test_qgen_010__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.code_completion.00` | v22_exemplar | hard · code_completion | |
| T2A-0734 | test | `test_qgen_010__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.code_completion.01` | v22_exemplar | hard · code_completion | |
| T2A-0735 | test | `test_qgen_010__qgen_exemplar` | qgen_exemplar | range_boundaries | `ex.range_boundaries.hard.code_completion.05` | v22_exemplar | hard · code_completion | |
| T2A-0736 | test | `test_qgen_010__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0737 | test | `test_qgen_010__qgen_curriculum` | qgen_curriculum | range_boundaries | `cur.range_boundaries.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0738 | test | `test_qgen_011__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.single_choice.08` | v22_exemplar | easy · single_choice | |
| T2A-0739 | test | `test_qgen_011__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.single_choice.06` | v22_exemplar | easy · single_choice | |
| T2A-0740 | test | `test_qgen_011__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.single_choice.00` | v22_exemplar | easy · single_choice | |
| T2A-0741 | test | `test_qgen_011__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.single_choice.16` | v22_exemplar | easy · single_choice | |
| T2A-0742 | test | `test_qgen_011__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.084e03e931fd` | episodic | Learner History · mistake | |
| T2A-0743 | test | `test_qgen_011__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.fd018b0295b8` | episodic | Learner History · assisted_attempt | |
| T2A-0744 | test | `test_qgen_012__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.single_choice.06` | v22_exemplar | hard · single_choice | |
| T2A-0745 | test | `test_qgen_012__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.single_choice.00` | v22_exemplar | hard · single_choice | |
| T2A-0746 | test | `test_qgen_012__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.single_choice.12` | v22_exemplar | hard · single_choice | |
| T2A-0747 | test | `test_qgen_012__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.single_choice.08` | v22_exemplar | hard · single_choice | |
| T2A-0748 | test | `test_qgen_012__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0749 | test | `test_qgen_012__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0750 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0751 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.01` | v22_exemplar | easy · multiple_select | |
| T2A-0752 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.00` | v22_exemplar | easy · multiple_select | |
| T2A-0753 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.05` | v22_exemplar | easy · multiple_select | |
| T2A-0754 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0755 | test | `test_qgen_013__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0756 | test | `test_qgen_013__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.fd018b0295b8` | episodic | Learner History · assisted_attempt | |
| T2A-0757 | test | `test_qgen_013__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.084e03e931fd` | episodic | Learner History · mistake | |
| T2A-0758 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.05` | v22_exemplar | medium · multiple_select | |
| T2A-0759 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.01` | v22_exemplar | medium · multiple_select | |
| T2A-0760 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-0761 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-0762 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-0763 | test | `test_qgen_014__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.multiple_select.00` | v22_exemplar | medium · multiple_select | |
| T2A-0764 | test | `test_qgen_014__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0765 | test | `test_qgen_014__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0766 | test | `test_qgen_015__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.output_prediction.06` | v22_exemplar | medium · output_prediction | |
| T2A-0767 | test | `test_qgen_015__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.output_prediction.15` | v22_exemplar | medium · output_prediction | |
| T2A-0768 | test | `test_qgen_015__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0769 | test | `test_qgen_015__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.output_prediction.09` | v22_exemplar | medium · output_prediction | |
| T2A-0770 | test | `test_qgen_015__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.output_prediction.00` | v22_exemplar | medium · output_prediction | |
| T2A-0771 | test | `test_qgen_015__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.fd018b0295b8` | episodic | Learner History · assisted_attempt | |
| T2A-0772 | test | `test_qgen_015__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.084e03e931fd` | episodic | Learner History · mistake | |
| T2A-0773 | test | `test_qgen_016__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.output_prediction.12` | v22_exemplar | hard · output_prediction | |
| T2A-0774 | test | `test_qgen_016__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.output_prediction.11` | v22_exemplar | hard · output_prediction | |
| T2A-0775 | test | `test_qgen_016__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.output_prediction.07` | v22_exemplar | hard · output_prediction | |
| T2A-0776 | test | `test_qgen_016__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.output_prediction.01` | v22_exemplar | hard · output_prediction | |
| T2A-0777 | test | `test_qgen_016__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.output_prediction.05` | v22_exemplar | hard · output_prediction | |
| T2A-0778 | test | `test_qgen_016__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0779 | test | `test_qgen_016__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0780 | test | `test_qgen_017__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.fill_in_the_blank.21` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0781 | test | `test_qgen_017__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.fill_in_the_blank.09` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0782 | test | `test_qgen_017__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.fill_in_the_blank.16` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0783 | test | `test_qgen_017__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0784 | test | `test_qgen_017__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.fill_in_the_blank.00` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0785 | test | `test_qgen_017__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.fd018b0295b8` | episodic | Learner History · assisted_attempt | |
| T2A-0786 | test | `test_qgen_017__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.084e03e931fd` | episodic | Learner History · mistake | |
| T2A-0787 | test | `test_qgen_018__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.fill_in_the_blank.00` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0788 | test | `test_qgen_018__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.fill_in_the_blank.06` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0789 | test | `test_qgen_018__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.fill_in_the_blank.09` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0790 | test | `test_qgen_018__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.fill_in_the_blank.16` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0791 | test | `test_qgen_018__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.hard.fill_in_the_blank.15` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0792 | test | `test_qgen_018__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0793 | test | `test_qgen_018__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0794 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-0795 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.05` | v22_exemplar | easy · code_completion | |
| T2A-0796 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.01` | v22_exemplar | easy · code_completion | |
| T2A-0797 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-0798 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.00` | v22_exemplar | easy · code_completion | |
| T2A-0799 | test | `test_qgen_019__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-0800 | test | `test_qgen_019__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.fd018b0295b8` | episodic | Learner History · assisted_attempt | |
| T2A-0801 | test | `test_qgen_019__qgen_curriculum` | qgen_curriculum | loop_tracing | `ep.test_qgen_profile_a.loop_tracing.084e03e931fd` | episodic | Learner History · mistake | |
| T2A-0802 | test | `test_qgen_020__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.code_completion.00` | v22_exemplar | medium · code_completion | |
| T2A-0803 | test | `test_qgen_020__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.code_completion.01` | v22_exemplar | medium · code_completion | |
| T2A-0804 | test | `test_qgen_020__qgen_exemplar` | qgen_exemplar | loop_tracing | `ex.loop_tracing.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-0805 | test | `test_qgen_020__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0806 | test | `test_qgen_020__qgen_curriculum` | qgen_curriculum | loop_tracing | `cur.loop_tracing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0807 | test | `test_qgen_021__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.single_choice.08` | v22_exemplar | easy · single_choice | |
| T2A-0808 | test | `test_qgen_021__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.single_choice.12` | v22_exemplar | easy · single_choice | |
| T2A-0809 | test | `test_qgen_021__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.single_choice.06` | v22_exemplar | easy · single_choice | |
| T2A-0810 | test | `test_qgen_021__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.single_choice.00` | v22_exemplar | easy · single_choice | |
| T2A-0811 | test | `test_qgen_021__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.67d6b1b75ffc` | episodic | Learner History · assisted_attempt | |
| T2A-0812 | test | `test_qgen_021__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.6b3df9278954` | episodic | Learner History · mistake | |
| T2A-0813 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.00` | v22_exemplar | medium · single_choice | |
| T2A-0814 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.01` | v22_exemplar | medium · single_choice | |
| T2A-0815 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.05` | v22_exemplar | medium · single_choice | |
| T2A-0816 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.06` | v22_exemplar | medium · single_choice | |
| T2A-0817 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.12` | v22_exemplar | medium · single_choice | |
| T2A-0818 | test | `test_qgen_022__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.single_choice.16` | v22_exemplar | medium · single_choice | |
| T2A-0819 | test | `test_qgen_022__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0820 | test | `test_qgen_022__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0821 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.01` | v22_exemplar | medium · multiple_select | |
| T2A-0822 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-0823 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.00` | v22_exemplar | medium · multiple_select | |
| T2A-0824 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.05` | v22_exemplar | medium · multiple_select | |
| T2A-0825 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-0826 | test | `test_qgen_023__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-0827 | test | `test_qgen_023__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.6b3df9278954` | episodic | Learner History · mistake | |
| T2A-0828 | test | `test_qgen_023__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.67d6b1b75ffc` | episodic | Learner History · assisted_attempt | |
| T2A-0829 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-0830 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.05` | v22_exemplar | hard · multiple_select | |
| T2A-0831 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-0832 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.00` | v22_exemplar | hard · multiple_select | |
| T2A-0833 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.01` | v22_exemplar | hard · multiple_select | |
| T2A-0834 | test | `test_qgen_024__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-0835 | test | `test_qgen_024__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0836 | test | `test_qgen_024__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0837 | test | `test_qgen_025__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.output_prediction.12` | v22_exemplar | easy · output_prediction | |
| T2A-0838 | test | `test_qgen_025__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.output_prediction.00` | v22_exemplar | easy · output_prediction | |
| T2A-0839 | test | `test_qgen_025__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.output_prediction.06` | v22_exemplar | easy · output_prediction | |
| T2A-0840 | test | `test_qgen_025__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.output_prediction.09` | v22_exemplar | easy · output_prediction | |
| T2A-0841 | test | `test_qgen_025__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.output_prediction.15` | v22_exemplar | easy · output_prediction | |
| T2A-0842 | test | `test_qgen_025__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.67d6b1b75ffc` | episodic | Learner History · assisted_attempt | |
| T2A-0843 | test | `test_qgen_025__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.6b3df9278954` | episodic | Learner History · mistake | |
| T2A-0844 | test | `test_qgen_026__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.output_prediction.15` | v22_exemplar | hard · output_prediction | |
| T2A-0845 | test | `test_qgen_026__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.output_prediction.06` | v22_exemplar | hard · output_prediction | |
| T2A-0846 | test | `test_qgen_026__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.output_prediction.00` | v22_exemplar | hard · output_prediction | |
| T2A-0847 | test | `test_qgen_026__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.output_prediction.09` | v22_exemplar | hard · output_prediction | |
| T2A-0848 | test | `test_qgen_026__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.output_prediction.12` | v22_exemplar | hard · output_prediction | |
| T2A-0849 | test | `test_qgen_026__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0850 | test | `test_qgen_026__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0851 | test | `test_qgen_027__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.fill_in_the_blank.09` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0852 | test | `test_qgen_027__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0853 | test | `test_qgen_027__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.fill_in_the_blank.00` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0854 | test | `test_qgen_027__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.fill_in_the_blank.21` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0855 | test | `test_qgen_027__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.easy.fill_in_the_blank.12` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0856 | test | `test_qgen_027__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.67d6b1b75ffc` | episodic | Learner History · assisted_attempt | |
| T2A-0857 | test | `test_qgen_027__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.6b3df9278954` | episodic | Learner History · mistake | |
| T2A-0858 | test | `test_qgen_028__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.fill_in_the_blank.15` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0859 | test | `test_qgen_028__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.fill_in_the_blank.09` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0860 | test | `test_qgen_028__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.fill_in_the_blank.00` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0861 | test | `test_qgen_028__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.fill_in_the_blank.06` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0862 | test | `test_qgen_028__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.fill_in_the_blank.21` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0863 | test | `test_qgen_028__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0864 | test | `test_qgen_028__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0865 | test | `test_qgen_029__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.code_completion.01` | v22_exemplar | medium · code_completion | |
| T2A-0866 | test | `test_qgen_029__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.code_completion.00` | v22_exemplar | medium · code_completion | |
| T2A-0867 | test | `test_qgen_029__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-0868 | test | `test_qgen_029__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.6b3df9278954` | episodic | Learner History · mistake | |
| T2A-0869 | test | `test_qgen_029__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `ep.test_qgen_profile_a.accumulator_initialization.67d6b1b75ffc` | episodic | Learner History · assisted_attempt | |
| T2A-0870 | test | `test_qgen_030__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.code_completion.05` | v22_exemplar | hard · code_completion | |
| T2A-0871 | test | `test_qgen_030__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.code_completion.01` | v22_exemplar | hard · code_completion | |
| T2A-0872 | test | `test_qgen_030__qgen_exemplar` | qgen_exemplar | accumulator_initialization | `ex.accumulator_initialization.hard.code_completion.00` | v22_exemplar | hard · code_completion | |
| T2A-0873 | test | `test_qgen_030__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0874 | test | `test_qgen_030__qgen_curriculum` | qgen_curriculum | accumulator_initialization | `cur.accumulator_initialization.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0875 | test | `test_qgen_031__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.single_choice.11` | v22_exemplar | medium · single_choice | |
| T2A-0876 | test | `test_qgen_031__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.single_choice.07` | v22_exemplar | medium · single_choice | |
| T2A-0877 | test | `test_qgen_031__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.single_choice.05` | v22_exemplar | medium · single_choice | |
| T2A-0878 | test | `test_qgen_031__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.single_choice.01` | v22_exemplar | medium · single_choice | |
| T2A-0879 | test | `test_qgen_031__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0880 | test | `test_qgen_031__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.ef7f57faf95e` | episodic | Learner History · assisted_attempt | |
| T2A-0881 | test | `test_qgen_031__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.c4ba74a26330` | episodic | Learner History · reveal | |
| T2A-0882 | test | `test_qgen_032__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.single_choice.15` | v22_exemplar | hard · single_choice | |
| T2A-0883 | test | `test_qgen_032__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.single_choice.07` | v22_exemplar | hard · single_choice | |
| T2A-0884 | test | `test_qgen_032__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.single_choice.01` | v22_exemplar | hard · single_choice | |
| T2A-0885 | test | `test_qgen_032__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.single_choice.05` | v22_exemplar | hard · single_choice | |
| T2A-0886 | test | `test_qgen_032__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0887 | test | `test_qgen_032__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0888 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0889 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.05` | v22_exemplar | easy · multiple_select | |
| T2A-0890 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0891 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0892 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.01` | v22_exemplar | easy · multiple_select | |
| T2A-0893 | test | `test_qgen_033__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.multiple_select.00` | v22_exemplar | easy · multiple_select | |
| T2A-0894 | test | `test_qgen_033__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0895 | test | `test_qgen_033__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.c4ba74a26330` | episodic | Learner History · reveal | |
| T2A-0896 | test | `test_qgen_033__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.ef7f57faf95e` | episodic | Learner History · assisted_attempt | |
| T2A-0897 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-0898 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-0899 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.01` | v22_exemplar | hard · multiple_select | |
| T2A-0900 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.05` | v22_exemplar | hard · multiple_select | |
| T2A-0901 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-0902 | test | `test_qgen_034__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.multiple_select.00` | v22_exemplar | hard · multiple_select | |
| T2A-0903 | test | `test_qgen_034__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0904 | test | `test_qgen_034__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0905 | test | `test_qgen_035__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.output_prediction.05` | v22_exemplar | easy · output_prediction | |
| T2A-0906 | test | `test_qgen_035__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.output_prediction.07` | v22_exemplar | easy · output_prediction | |
| T2A-0907 | test | `test_qgen_035__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.output_prediction.01` | v22_exemplar | easy · output_prediction | |
| T2A-0908 | test | `test_qgen_035__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.output_prediction.10` | v22_exemplar | easy · output_prediction | |
| T2A-0909 | test | `test_qgen_035__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0910 | test | `test_qgen_035__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.ef7f57faf95e` | episodic | Learner History · assisted_attempt | |
| T2A-0911 | test | `test_qgen_035__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.c4ba74a26330` | episodic | Learner History · reveal | |
| T2A-0912 | test | `test_qgen_036__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.output_prediction.01` | v22_exemplar | medium · output_prediction | |
| T2A-0913 | test | `test_qgen_036__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.output_prediction.07` | v22_exemplar | medium · output_prediction | |
| T2A-0914 | test | `test_qgen_036__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.output_prediction.11` | v22_exemplar | medium · output_prediction | |
| T2A-0915 | test | `test_qgen_036__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0916 | test | `test_qgen_036__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.output_prediction.05` | v22_exemplar | medium · output_prediction | |
| T2A-0917 | test | `test_qgen_036__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0918 | test | `test_qgen_036__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0919 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.09` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0920 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.00` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0921 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.10` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0922 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.16` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0923 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.01` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0924 | test | `test_qgen_037__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.medium.fill_in_the_blank.06` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-0925 | test | `test_qgen_037__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.ef7f57faf95e` | episodic | Learner History · assisted_attempt | |
| T2A-0926 | test | `test_qgen_037__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.c4ba74a26330` | episodic | Learner History · reveal | |
| T2A-0927 | test | `test_qgen_037__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0928 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.01` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0929 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.06` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0930 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.00` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0931 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.16` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0932 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.10` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0933 | test | `test_qgen_038__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.fill_in_the_blank.09` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0934 | test | `test_qgen_038__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0935 | test | `test_qgen_038__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0936 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-0937 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-0938 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.01` | v22_exemplar | easy · code_completion | |
| T2A-0939 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.00` | v22_exemplar | easy · code_completion | |
| T2A-0940 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.05` | v22_exemplar | easy · code_completion | |
| T2A-0941 | test | `test_qgen_039__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-0942 | test | `test_qgen_039__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0943 | test | `test_qgen_039__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.ef7f57faf95e` | episodic | Learner History · assisted_attempt | |
| T2A-0944 | test | `test_qgen_039__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `ep.test_qgen_profile_a.conditional_evaluation.c4ba74a26330` | episodic | Learner History · reveal | |
| T2A-0945 | test | `test_qgen_040__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.code_completion.00` | v22_exemplar | hard · code_completion | |
| T2A-0946 | test | `test_qgen_040__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.code_completion.05` | v22_exemplar | hard · code_completion | |
| T2A-0947 | test | `test_qgen_040__qgen_exemplar` | qgen_exemplar | conditional_evaluation | `ex.conditional_evaluation.hard.code_completion.01` | v22_exemplar | hard · code_completion | |
| T2A-0948 | test | `test_qgen_040__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0949 | test | `test_qgen_040__qgen_curriculum` | qgen_curriculum | conditional_evaluation | `cur.conditional_evaluation.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0950 | test | `test_qgen_041__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.single_choice.00` | v22_exemplar | easy · single_choice | |
| T2A-0951 | test | `test_qgen_041__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.single_choice.06` | v22_exemplar | easy · single_choice | |
| T2A-0952 | test | `test_qgen_041__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.single_choice.16` | v22_exemplar | easy · single_choice | |
| T2A-0953 | test | `test_qgen_041__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.48a00f523686` | episodic | Learner History · assisted_attempt | |
| T2A-0954 | test | `test_qgen_041__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.ed0c34527330` | episodic | Learner History · mistake | |
| T2A-0955 | test | `test_qgen_042__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.single_choice.08` | v22_exemplar | hard · single_choice | |
| T2A-0956 | test | `test_qgen_042__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.single_choice.22` | v22_exemplar | hard · single_choice | |
| T2A-0957 | test | `test_qgen_042__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.single_choice.16` | v22_exemplar | hard · single_choice | |
| T2A-0958 | test | `test_qgen_042__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0959 | test | `test_qgen_042__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0960 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.11` | v22_exemplar | easy · multiple_select | |
| T2A-0961 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.01` | v22_exemplar | easy · multiple_select | |
| T2A-0962 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.00` | v22_exemplar | easy · multiple_select | |
| T2A-0963 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.12` | v22_exemplar | easy · multiple_select | |
| T2A-0964 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.05` | v22_exemplar | easy · multiple_select | |
| T2A-0965 | test | `test_qgen_043__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.multiple_select.10` | v22_exemplar | easy · multiple_select | |
| T2A-0966 | test | `test_qgen_043__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.ed0c34527330` | episodic | Learner History · mistake | |
| T2A-0967 | test | `test_qgen_043__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.48a00f523686` | episodic | Learner History · assisted_attempt | |
| T2A-0968 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-0969 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-0970 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.05` | v22_exemplar | medium · multiple_select | |
| T2A-0971 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-0972 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.00` | v22_exemplar | medium · multiple_select | |
| T2A-0973 | test | `test_qgen_044__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.multiple_select.01` | v22_exemplar | medium · multiple_select | |
| T2A-0974 | test | `test_qgen_044__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0975 | test | `test_qgen_044__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0976 | test | `test_qgen_045__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.output_prediction.12` | v22_exemplar | medium · output_prediction | |
| T2A-0977 | test | `test_qgen_045__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.output_prediction.15` | v22_exemplar | medium · output_prediction | |
| T2A-0978 | test | `test_qgen_045__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.output_prediction.00` | v22_exemplar | medium · output_prediction | |
| T2A-0979 | test | `test_qgen_045__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.output_prediction.09` | v22_exemplar | medium · output_prediction | |
| T2A-0980 | test | `test_qgen_045__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.output_prediction.06` | v22_exemplar | medium · output_prediction | |
| T2A-0981 | test | `test_qgen_045__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.48a00f523686` | episodic | Learner History · assisted_attempt | |
| T2A-0982 | test | `test_qgen_045__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.ed0c34527330` | episodic | Learner History · mistake | |
| T2A-0983 | test | `test_qgen_046__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.output_prediction.21` | v22_exemplar | hard · output_prediction | |
| T2A-0984 | test | `test_qgen_046__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.output_prediction.09` | v22_exemplar | hard · output_prediction | |
| T2A-0985 | test | `test_qgen_046__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.output_prediction.18` | v22_exemplar | hard · output_prediction | |
| T2A-0986 | test | `test_qgen_046__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.output_prediction.11` | v22_exemplar | hard · output_prediction | |
| T2A-0987 | test | `test_qgen_046__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-0988 | test | `test_qgen_046__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-0989 | test | `test_qgen_047__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.fill_in_the_blank.21` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0990 | test | `test_qgen_047__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.fill_in_the_blank.11` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0991 | test | `test_qgen_047__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.fill_in_the_blank.01` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0992 | test | `test_qgen_047__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0993 | test | `test_qgen_047__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.fill_in_the_blank.16` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-0994 | test | `test_qgen_047__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.48a00f523686` | episodic | Learner History · assisted_attempt | |
| T2A-0995 | test | `test_qgen_047__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.ed0c34527330` | episodic | Learner History · mistake | |
| T2A-0996 | test | `test_qgen_048__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.fill_in_the_blank.10` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0997 | test | `test_qgen_048__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.fill_in_the_blank.17` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0998 | test | `test_qgen_048__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.fill_in_the_blank.16` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-0999 | test | `test_qgen_048__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.hard.fill_in_the_blank.23` | v22_exemplar | hard · fill_in_the_blank | |
| T2A-1000 | test | `test_qgen_048__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1001 | test | `test_qgen_048__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1002 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.10` | v22_exemplar | easy · code_completion | |
| T2A-1003 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.05` | v22_exemplar | easy · code_completion | |
| T2A-1004 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.11` | v22_exemplar | easy · code_completion | |
| T2A-1005 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.00` | v22_exemplar | easy · code_completion | |
| T2A-1006 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.01` | v22_exemplar | easy · code_completion | |
| T2A-1007 | test | `test_qgen_049__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.easy.code_completion.12` | v22_exemplar | easy · code_completion | |
| T2A-1008 | test | `test_qgen_049__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.48a00f523686` | episodic | Learner History · assisted_attempt | |
| T2A-1009 | test | `test_qgen_049__qgen_curriculum` | qgen_curriculum | list_indexing | `ep.test_qgen_profile_a.list_indexing.ed0c34527330` | episodic | Learner History · mistake | |
| T2A-1010 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.00` | v22_exemplar | medium · code_completion | |
| T2A-1011 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.10` | v22_exemplar | medium · code_completion | |
| T2A-1012 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-1013 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.01` | v22_exemplar | medium · code_completion | |
| T2A-1014 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.12` | v22_exemplar | medium · code_completion | |
| T2A-1015 | test | `test_qgen_050__qgen_exemplar` | qgen_exemplar | list_indexing | `ex.list_indexing.medium.code_completion.11` | v22_exemplar | medium · code_completion | |
| T2A-1016 | test | `test_qgen_050__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1017 | test | `test_qgen_050__qgen_curriculum` | qgen_curriculum | list_indexing | `cur.list_indexing.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1018 | test | `test_qgen_051__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.single_choice.07` | v22_exemplar | easy · single_choice | |
| T2A-1019 | test | `test_qgen_051__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.single_choice.01` | v22_exemplar | easy · single_choice | |
| T2A-1020 | test | `test_qgen_051__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.single_choice.11` | v22_exemplar | easy · single_choice | |
| T2A-1021 | test | `test_qgen_051__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.single_choice.05` | v22_exemplar | easy · single_choice | |
| T2A-1022 | test | `test_qgen_051__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e0e41962e3` | episodic | Learner History · reveal | |
| T2A-1023 | test | `test_qgen_051__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.d5bea9a37abb` | episodic | Learner History · mistake | |
| T2A-1024 | test | `test_qgen_051__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e5137c78fd` | episodic | Learner History · assisted_attempt | |
| T2A-1025 | test | `test_qgen_052__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.single_choice.12` | v22_exemplar | medium · single_choice | |
| T2A-1026 | test | `test_qgen_052__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.single_choice.05` | v22_exemplar | medium · single_choice | |
| T2A-1027 | test | `test_qgen_052__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.single_choice.07` | v22_exemplar | medium · single_choice | |
| T2A-1028 | test | `test_qgen_052__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.single_choice.01` | v22_exemplar | medium · single_choice | |
| T2A-1029 | test | `test_qgen_052__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.single_choice.11` | v22_exemplar | medium · single_choice | |
| T2A-1030 | test | `test_qgen_052__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1031 | test | `test_qgen_052__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1032 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.00` | v22_exemplar | medium · multiple_select | |
| T2A-1033 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.10` | v22_exemplar | medium · multiple_select | |
| T2A-1034 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.05` | v22_exemplar | medium · multiple_select | |
| T2A-1035 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.01` | v22_exemplar | medium · multiple_select | |
| T2A-1036 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.12` | v22_exemplar | medium · multiple_select | |
| T2A-1037 | test | `test_qgen_053__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.multiple_select.11` | v22_exemplar | medium · multiple_select | |
| T2A-1038 | test | `test_qgen_053__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e0e41962e3` | episodic | Learner History · reveal | |
| T2A-1039 | test | `test_qgen_053__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e5137c78fd` | episodic | Learner History · assisted_attempt | |
| T2A-1040 | test | `test_qgen_053__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.d5bea9a37abb` | episodic | Learner History · mistake | |
| T2A-1041 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.01` | v22_exemplar | hard · multiple_select | |
| T2A-1042 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.05` | v22_exemplar | hard · multiple_select | |
| T2A-1043 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.12` | v22_exemplar | hard · multiple_select | |
| T2A-1044 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.00` | v22_exemplar | hard · multiple_select | |
| T2A-1045 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.10` | v22_exemplar | hard · multiple_select | |
| T2A-1046 | test | `test_qgen_054__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.multiple_select.11` | v22_exemplar | hard · multiple_select | |
| T2A-1047 | test | `test_qgen_054__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1048 | test | `test_qgen_054__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1049 | test | `test_qgen_055__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.output_prediction.00` | v22_exemplar | easy · output_prediction | |
| T2A-1050 | test | `test_qgen_055__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.output_prediction.12` | v22_exemplar | easy · output_prediction | |
| T2A-1051 | test | `test_qgen_055__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.output_prediction.06` | v22_exemplar | easy · output_prediction | |
| T2A-1052 | test | `test_qgen_055__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.output_prediction.09` | v22_exemplar | easy · output_prediction | |
| T2A-1053 | test | `test_qgen_055__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.d5bea9a37abb` | episodic | Learner History · mistake | |
| T2A-1054 | test | `test_qgen_055__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e5137c78fd` | episodic | Learner History · assisted_attempt | |
| T2A-1055 | test | `test_qgen_055__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e0e41962e3` | episodic | Learner History · reveal | |
| T2A-1056 | test | `test_qgen_056__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.output_prediction.18` | v22_exemplar | hard · output_prediction | |
| T2A-1057 | test | `test_qgen_056__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.output_prediction.00` | v22_exemplar | hard · output_prediction | |
| T2A-1058 | test | `test_qgen_056__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.output_prediction.12` | v22_exemplar | hard · output_prediction | |
| T2A-1059 | test | `test_qgen_056__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.output_prediction.09` | v22_exemplar | hard · output_prediction | |
| T2A-1060 | test | `test_qgen_056__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.output_prediction.06` | v22_exemplar | hard · output_prediction | |
| T2A-1061 | test | `test_qgen_056__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1062 | test | `test_qgen_056__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1063 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.00` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1064 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.07` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1065 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.09` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1066 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.06` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1067 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.16` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1068 | test | `test_qgen_057__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.easy.fill_in_the_blank.01` | v22_exemplar | easy · fill_in_the_blank | |
| T2A-1069 | test | `test_qgen_057__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.d5bea9a37abb` | episodic | Learner History · mistake | |
| T2A-1070 | test | `test_qgen_057__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e0e41962e3` | episodic | Learner History · reveal | |
| T2A-1071 | test | `test_qgen_057__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e5137c78fd` | episodic | Learner History · assisted_attempt | |
| T2A-1072 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.00` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1073 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.09` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1074 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.06` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1075 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.16` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1076 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.10` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1077 | test | `test_qgen_058__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.fill_in_the_blank.01` | v22_exemplar | medium · fill_in_the_blank | |
| T2A-1078 | test | `test_qgen_058__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1079 | test | `test_qgen_058__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
| T2A-1080 | test | `test_qgen_059__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.code_completion.05` | v22_exemplar | medium · code_completion | |
| T2A-1081 | test | `test_qgen_059__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.code_completion.00` | v22_exemplar | medium · code_completion | |
| T2A-1082 | test | `test_qgen_059__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.medium.code_completion.01` | v22_exemplar | medium · code_completion | |
| T2A-1083 | test | `test_qgen_059__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e0e41962e3` | episodic | Learner History · reveal | |
| T2A-1084 | test | `test_qgen_059__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.d5bea9a37abb` | episodic | Learner History · mistake | |
| T2A-1085 | test | `test_qgen_059__qgen_curriculum` | qgen_curriculum | function_returns | `ep.test_qgen_profile_a.function_returns.66e5137c78fd` | episodic | Learner History · assisted_attempt | |
| T2A-1086 | test | `test_qgen_060__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.code_completion.01` | v22_exemplar | hard · code_completion | |
| T2A-1087 | test | `test_qgen_060__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.code_completion.05` | v22_exemplar | hard · code_completion | |
| T2A-1088 | test | `test_qgen_060__qgen_exemplar` | qgen_exemplar | function_returns | `ex.function_returns.hard.code_completion.00` | v22_exemplar | hard · code_completion | |
| T2A-1089 | test | `test_qgen_060__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.3.0` | curriculum | Difficulty Progression and Worked Examples | |
| T2A-1090 | test | `test_qgen_060__qgen_curriculum` | qgen_curriculum | function_returns | `cur.function_returns.2.0` | curriculum | Common Misconceptions and Pitfalls | |
