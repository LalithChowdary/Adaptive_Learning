"""Canonical IEEE reference registry and References section builder."""
from docx.shared import Pt
from .style import add_h1, add_p

# Ordered deduplicated IEEE references [1]..[32].
# [1]..[15] preserve the exact 15 baseline references from
# outputs/Adaptive_Tutoring_Report_With_Advisor_Signature.docx and
# outputs/report_revision_work/build_revised_report.py.
# [16]..[32] cover the cognitive memory, spaced repetition, IRT, RL, and
# Hybrid RAG literature grounded in 3/ and PROJECT.md.
REFERENCE_LIST = [
    (
        "ruan2024",
        'S. Ruan et al., "Reinforcement learning tutor better supported lower performers in a math task," Machine Learning, vol. 113, pp. 3023-3048, 2024, doi: 10.1007/s10994-023-06423-9. [Online]. Available: https://link.springer.com/article/10.1007/s10994-023-06423-9',
    ),
    (
        "nam2024",
        'H. Nam, O. Gottesman, A. Zhang, D. Foster, E. Brunskill, and L. Ungar, "Efficient RL for optimizing conversation level outcomes with an LLM-based tutor," arXiv:2507.16252, 2025. [Online]. Available: https://arxiv.org/abs/2507.16252',
    ),
    (
        "schulman2017ppo",
        'J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347',
    ),
    (
        "piech2015dkt",
        'C. Piech et al., "Deep knowledge tracing," in Advances in Neural Information Processing Systems, vol. 28, 2015. [Online]. Available: https://arxiv.org/abs/1506.05908',
    ),
    (
        "zhang2017dkvmn",
        'J. Zhang, X. Shi, I. King, and D.-Y. Yeung, "Dynamic key-value memory networks for knowledge tracing," in Proc. 26th International Conference on World Wide Web (WWW), pp. 765-774, 2017, doi: 10.1145/3038912.3052580. [Online]. Available: https://archives.iw3c2.org/www2017/proceedings/proceedings/p765.pdf',
    ),
    (
        "pandey2019sakt",
        'S. Pandey and G. Karypis, "A self-attentive model for knowledge tracing," in Proc. 12th International Conference on Educational Data Mining (EDM), pp. 384-389, 2019. [Online]. Available: https://arxiv.org/abs/1907.06837',
    ),
    (
        "settles2016hlr",
        'B. Settles and B. Meeder, "A trainable spaced repetition model for language learning," in Proc. 54th Annual Meeting of the Association for Computational Linguistics (ACL), vol. 1, pp. 1848-1858, 2016, doi: 10.18653/v1/P16-1174. [Online]. Available: https://aclanthology.org/P16-1174/',
    ),
    (
        "dgkt",
        'Y. Xie, S. Peng, W. Yang, M. Yang, and Y. Gao, "Domain generalizable knowledge tracing via concept aggregation and relation-based attention," arXiv:2407.02547, ver. 2, 2025. [Online]. Available: https://arxiv.org/abs/2407.02547v2',
    ),
    (
        "scarlatos2024",
        'A. Scarlatos, N. Liu, J. Lee, R. Baraniuk, and A. Lan, "Training LLM-based tutors to improve student learning outcomes in dialogues," in Artificial Intelligence in Education (AIED), LNCS, vol. 15877, 2025, doi: 10.1007/978-3-031-98414-3_18. [Online]. Available: https://arxiv.org/abs/2503.06424',
    ),
    (
        "cola",
        'C. Jia et al., "Controlling large language model with latent action," in Proc. 42nd International Conference on Machine Learning (ICML), PMLR, vol. 267, pp. 27331-27372, 2025. [Online]. Available: https://proceedings.mlr.press/v267/jia25e.html',
    ),
    (
        "lewis2020rag",
        'P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459-9474, 2020. [Online]. Available: https://arxiv.org/abs/2005.11401',
    ),
    (
        "typesafe1",
        'TypeSafe, "Score," TypeSafe Documentation. Accessed: Oct. 14, 2026. [Online]. Available: https://docs.typesafe.ai/primitives/score',
    ),
    (
        "typesafe2",
        'TypeSafe, "Confidence," TypeSafe Documentation. Accessed: Oct. 14, 2026. [Online]. Available: https://docs.typesafe.ai/confidence',
    ),
    (
        "assistments2012",
        'ASSISTments, "2012-13 school data with affect," ASSISTments Data. Accessed: Oct. 14, 2026. [Online]. Available: https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect',
    ),
    (
        "hu2022lora",
        'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. International Conference on Learning Representations (ICLR), 2022. [Online]. Available: https://arxiv.org/abs/2106.09685',
    ),
    (
        "corbett1995bkt",
        'A. T. Corbett and J. R. Anderson, "Knowledge tracing: Modeling the acquisition of procedural knowledge," User Modeling and User-Adapted Interaction, vol. 4, no. 4, pp. 253-278, 1995, doi: 10.1007/BF01099821.',
    ),
    (
        "ghosh2020akt",
        'A. Ghosh, N. Heffernan, and A. S. Lan, "Context-aware attentive knowledge tracing," in Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), pp. 2330-2339, 2020, doi: 10.1145/3394486.3403282.',
    ),
    (
        "choi2020saint",
        'Y. Choi, Y. Lee, J. Cho, J. Baek, B. Kim, Y. Cha, D. Shin, C. Bae, and J. Heo, "Towards an appropriate query, key, and value computation for knowledge tracing," in Proc. 7th ACM Conference on Learning @ Scale (L@S), pp. 341-344, 2020, doi: 10.1145/3386527.3405945.',
    ),
    (
        "bahrick1984permastore",
        'H. P. Bahrick, "Semantic memory content in permastore: Fifty years of memory for Spanish learned in school," Journal of Experimental Psychology: General, vol. 113, no. 1, pp. 1-29, 1984, doi: 10.1037/0096-3445.113.1.1.',
    ),
    (
        "wickelgren1974",
        'W. A. Wickelgren, "Single-trace fragility theory of memory dynamics," Memory & Cognition, vol. 2, no. 4, pp. 775-780, 1974, doi: 10.3758/BF03198154.',
    ),
    (
        "rubin1996",
        'D. C. Rubin and A. E. Wenzel, "One hundred years of forgetting: A quantitative description of retention," Psychological Review, vol. 103, no. 4, pp. 734-760, 1996, doi: 10.1037/0033-295X.103.4.734.',
    ),
    (
        "pavlik2005actr",
        'P. I. Pavlik Jr. and J. R. Anderson, "Practice and forgetting effects on vocabulary memory: An activation-based model of the spacing effect," Cognitive Science, vol. 29, no. 4, pp. 559-586, 2005, doi: 10.1207/s15516709cog0000_14.',
    ),
    (
        "ye2022fsrs",
        'J. Ye et al., "A stochastic shortest path algorithm for optimizing spaced repetition scheduling," in Proc. 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 4381-4390, 2022, doi: 10.1145/3534678.3539081.',
    ),
    (
        "birnbaum1968irt",
        'A. Birnbaum, "Some latent trait models and their use in inferring an examinee\'s ability," in Statistical Theories of Mental Test Scores, F. M. Lord and M. R. Novick, Eds. Reading, MA, USA: Addison-Wesley, 1968, pp. 397-479.',
    ),
    (
        "wise2005rte",
        'S. L. Wise and X. Kong, "Response time effort: A new measure of examinee motivation in computer-based tests," Applied Measurement in Education, vol. 18, no. 2, pp. 163-183, 2005, doi: 10.1207/s15324818ame1802_2.',
    ),
    (
        "schulman2016gae",
        'J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in Proc. International Conference on Learning Representations (ICLR), 2016. [Online]. Available: https://arxiv.org/abs/1506.02438',
    ),
    (
        "robertson2009bm25",
        'S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333-389, 2009, doi: 10.1561/1500000019.',
    ),
    (
        "cormack2009rrf",
        'G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms condorcet and individual rank learning methods," in Proc. 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pp. 758-759, 2009, doi: 10.1145/1571941.1572114.',
    ),
    (
        "che2018grud",
        'Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent neural networks for multivariate time series with missing values," Scientific Reports, vol. 8, no. 1, Art. no. 6085, 2018, doi: 10.1038/s41598-018-24271-9.',
    ),
    (
        "cho2014gru",
        'K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734, 2014, doi: 10.3115/v1/D14-1179.',
    ),
    (
        "vaswani2017attention",
        'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5998-6008, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762',
    ),
    (
        "mozer2016dash",
        'M. C. Mozer and R. V. Lindsey, "Predicting and improving memory retention: Psychological theory matters in the big data era," in Big Data in Cognitive Science, M. N. Jones, Ed. New York, NY, USA: Psychology Press, 2016, pp. 43-73.',
    ),
]

_CANONICAL_INDEX = {key: idx for idx, (key, _) in enumerate(REFERENCE_LIST, start=1)}

_ALIASES = {
    "baseline14": "assistments2012",
    "assistments": "assistments2012",
    "baseline15": "hu2022lora",
    "lora": "hu2022lora",
    "ruan": "ruan2024",
    "nam": "nam2024",
    "nam2025": "nam2024",
    "ppo": "schulman2017ppo",
    "dkt": "piech2015dkt",
    "dkvmn": "zhang2017dkvmn",
    "sakt": "pandey2019sakt",
    "settles": "settles2016hlr",
    "scarlatos": "scarlatos2024",
    "scarlatos2025": "scarlatos2024",
    "rag": "lewis2020rag",
    "jevscore": "typesafe1",
    "jevconfidence": "typesafe2",
}


def cite(key: str) -> str:
    """Return 1-indexed IEEE citation tag '[N]' for key or raise KeyError."""
    canonical = _ALIASES.get(key, key)
    if canonical not in _CANONICAL_INDEX:
        raise KeyError(f"Unknown citation key: {key!r}")
    return f"[{_CANONICAL_INDEX[canonical]}]"


def cite_many(*keys: str) -> str:
    """Return joined IEEE citation tags like '[1], [3]' or '[1]-[3]'."""
    return ", ".join(cite(k) for k in keys)


def build(doc, ctx=None):
    """Append the IEEE References section to `doc`."""
    add_h1(doc, "References", page_break=True)
    for idx, (_, citation) in enumerate(REFERENCE_LIST, start=1):
        p = add_p(
            doc,
            f"[{idx}] {citation}",
            size=10.5,
            before=0,
            after=5,
            line_spacing=1.25,
        )
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-24)
        p.paragraph_format.keep_together = True
