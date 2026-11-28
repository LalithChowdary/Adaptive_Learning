"""Chapter 2 — Literature Survey."""
from .references import cite
from .style import add_bullet, add_h1, add_h2, add_p, add_table


def build(doc, ctx=None):
    """Append Chapter 2 — Literature Survey to `doc`."""
    add_h1(doc, "Chapter 2 — Literature Survey", page_break=True)

    add_p(
        doc,
        "Building an effective AI Python tutor requires solving four connected problems: "
        "(1) tracking what a student still remembers after days or weeks away, "
        "(2) separating genuine understanding from lucky guesses or rushed clicks, "
        "(3) choosing the best next teaching step under a limited study budget, and "
        "(4) generating accurate, age- and skill-appropriate explanations without leaking quiz answers. "
        "This chapter introduces the existing research behind each piece in plain English, explaining "
        "the intuition first before giving the standard technical names.",
    )

    # ------------------------------------------------------------------
    # 2.1 Knowledge Tracing
    # ------------------------------------------------------------------
    add_h2(doc, "2.1 Knowledge Tracing: From Rule-Based Mastery to Deep Sequences")

    add_p(
        doc,
        "Before a tutor can pick a helpful exercise, it needs a way to answer a simple question: "
        "\"If I give this student a Python problem right now, how likely are they to solve it on their own?\" "
        "In educational AI, estimating a student's hidden skill level from their past answers is called "
        "knowledge tracing.",
    )

    add_p(
        doc,
        f"The earliest widely used method is Bayesian Knowledge Tracing (BKT) {cite('corbett1995bkt')}. "
        "Imagine a light switch inside the student's head for each skill: the student either has not learned "
        "the rule yet or has mastered it. BKT updates the chance that the switch is \"on\" after every attempt "
        "using four probabilities: starting knowledge, the chance of learning after a try, the chance of "
        "making a careless slip even when mastered, and the chance of guessing right by luck. "
        "Importantly, BKT treats correct and incorrect answers differently (outcome-signed evidence). "
        "However, standard BKT assumes every question on a skill has the exact same difficulty and ignores "
        "how much time passes between study sessions.",
    )

    add_p(
        doc,
        f"Psychometrics offers a cleaner way to handle question difficulty through Three-Parameter Logistic "
        f"Item Response Theory (3PL IRT) {cite('birnbaum1968irt')}. Rather than treating all problems as equal, "
        "3PL IRT predicts pass probability using three intuitive properties of a question: "
        "(1) difficulty (how strong a student must be to have a fair chance), "
        "(2) discrimination (how sharply the question separates beginners from experts), and "
        "(3) a guessing floor (for example, a 4-option multiple-choice question has a 25% floor even for random clicking). "
        "Classical 3PL IRT captures a static snapshot of ability, so it does not model how skill changes across time.",
    )

    add_p(
        doc,
        f"To learn richer patterns directly from student logs, Deep Knowledge Tracing (DKT) {cite('piech2015dkt')} "
        "replaced hand-written rules with a recurrent neural network (such as an LSTM or a Gated Recurrent Unit, "
        f"known as a GRU {cite('cho2014gru')}). A recurrent network reads a student's attempts one by one in order "
        "and keeps a compact running summary vector—like a mental notepad—that gets updated at each step. "
        f"Dynamic Key-Value Memory Networks (DKVMN) {cite('zhang2017dkvmn')} extended this idea by storing fixed "
        "concept \"keys\" alongside changing \"value\" slots so different skills do not overwrite one another as easily.",
    )

    add_p(
        doc,
        f"Later models adopted the Transformer attention mechanism {cite('vaswani2017attention')}, which lets a "
        "model look back at any earlier attempt directly rather than squeezing everything through one sequential "
        f"notepad. Self-Attentive Knowledge Tracing (SAKT) {cite('pandey2019sakt')} applies causal self-attention "
        f"to student interaction logs. Context-Aware Attentive Knowledge Tracing (AKT) {cite('ghosh2020akt')} adds "
        f"exponential distance decay in attention weights, and SAINT {cite('choi2020saint')} separates exercise "
        "streams from response streams in an encoder-decoder Transformer. Meanwhile, Domain Generalizable Knowledge "
        f"Tracing (DGKT) {cite('dgkt')} studies how concept aggregation and relation-based attention help models "
        f"transfer across subjects, and GRU-D {cite('che2018grud')} handles irregular time gaps by exponentially "
        "decaying missing input features toward empirical baselines and decaying hidden states toward zero over time.",
    )

    add_p(
        doc,
        "Despite strong scores on short-term benchmarks, standard next-step accuracy models suffer from three "
        "practical weaknesses when deployed in a real long-horizon tutor:",
    )

    add_bullet(
        doc,
        "Short-session bias and question-ID lock-in: ",
        f"Large public logs such as ASSISTments 2012-2013 {cite('assistments2012')} focus on immediate next-question "
        "prediction rather than independent 7-day or 30-day retention tests. In our own audit of 6.12 million "
        "ASSISTments rows, 55.71% of records were missing a canonical skill tag, forcing models to memorize 117,966 "
        "problem IDs (118,232 total item, skill, and special tokens) that cannot transfer to new Python exercises.",
    )
    add_bullet(
        doc,
        "Unconstrained time-decay and rapid-guess bugs: ",
        "If an unconstrained neural network sees fast response times during training, it may assume speed always "
        "means high skill—even when a student clicks randomly in 6 seconds and scores 0%. Similarly, unconstrained "
        "hidden-state decay toward zero or a learned population prior (as in GRU-D-style models or time_decay_gru) "
        "can accidentally make predicted skill drift upward during a break after a student fails.",
    )
    add_bullet(
        doc,
        "Sequence replay cost vs. per-skill state: ",
        "Transformers like SAKT and AKT were designed to pick relevant past events out of long interleaved logs "
        "containing dozens of mixed topics, at a cost that grows with history length L. When records are cleanly "
        "isolated per (learner, skill) pair—as verified in our Stage 2 ASSISTments experiment where recurrent "
        "LSTM-DKT outperformed SAKT-Transformer—a structured O(1) recurrent state per skill is both faster and "
        "simpler to feed into a downstream teaching policy.",
    )

    # ------------------------------------------------------------------
    # 2.2 Human Memory, Spaced Repetition, and Long-Term Retention
    # ------------------------------------------------------------------
    add_h2(doc, "2.2 Human Memory, Spaced Repetition, and Long-Term Retention")

    add_p(
        doc,
        "Everyone has experienced passing a quiz right after cramming, only to blank out two weeks later. "
        "Cognitive psychology has studied how human memory actually fades—and how spacing practice across days "
        "turns fragile short-term fluency into lasting knowledge.",
    )

    add_p(
        doc,
        f"Early mathematical models assumed memory fades at a constant percentage per day (exponential decay), "
        f"which eventually plunges toward zero. However, Wickelgren (1974) {cite('wickelgren1974')} and a massive "
        f"meta-analysis of 100 years of forgetting data by Rubin and Wenzel (1996) {cite('rubin1996')} showed that "
        "human forgetting usually follows a power law: memory drops fast right after study, then flattens out "
        "into a much slower tail over weeks and months.",
    )

    add_p(
        doc,
        f"Even more striking, Bahrick (1984) {cite('bahrick1984permastore')} tracked people who studied Spanish in "
        "school for up to 50 years and discovered what he called Permastore: once knowledge is deeply consolidated, "
        "a durable baseline floor remains intact for decades rather than decaying to zero. Cognitive architectures "
        f"such as ACT-R {cite('pavlik2005actr')} and DASH {cite('mozer2016dash')} formalized how spaced practice "
        "episodes build stronger long-term activation than back-to-back cramming.",
    )

    add_p(
        doc,
        f"Spaced repetition schedulers bring these findings into software. Duolingo's Half-Life Regression (HLR) "
        f"{cite('settles2016hlr')} predicts recall probability using an explicit memory half-life h_t (measured in "
        "days—the time until recall drops to 50% of its initial value). Every successful recall lengthens h_t, while "
        f"failures shorten it. More recently, the Free Spaced Repetition Scheduler (FSRS) {cite('ye2022fsrs')} "
        "paired power-law forgetting with stochastic shortest-path optimization to choose review intervals that "
        "minimize total study effort.",
    )

    add_p(
        doc,
        f"Finally, Wise and Kong (2005) {cite('wise2005rte')} introduced Response Time Effort (RTE) in computer-based "
        "testing, demonstrating that unusually fast answers indicate disengaged rapid guessing rather than true "
        "mastery. Combining HLR half-lives, power-law tails, Bahrick's Permastore floor, 3PL IRT difficulty, and "
        "rapid-guess filtering prevents a memory tracker from overestimating a student who rushes or crams.",
    )

    # ------------------------------------------------------------------
    # 2.3 Reinforcement Learning for Adaptive Tutoring
    # ------------------------------------------------------------------
    add_h2(doc, "2.3 Reinforcement Learning for Adaptive Tutoring")

    add_p(
        doc,
        "Tracking memory tells the system what the student knows right now, but it does not say what the tutor "
        "should do next. Should it explain the concept first, show a worked example, ask an easier question, "
        "challenge the student with a harder problem, ask them to explain their reasoning, or jump back to fix a "
        "forgotten prerequisite? Reinforcement Learning (RL) provides a principled way to learn this decision strategy.",
    )

    add_p(
        doc,
        "In RL, tutoring is modeled as a Markov Decision Process (MDP). At each turn t, the policy observes a "
        "state summary s_t of the learner, picks one teaching action a_t, receives a reward r_t reflecting learning "
        "progress, and transitions to a new state after the student responds. Unlike static if/else rules, an RL "
        "policy plans several steps ahead so actions that cost a little extra time now can pay off weeks later.",
    )

    add_p(
        doc,
        f"A widely used RL algorithm for stable training is Proximal Policy Optimization (PPO) "
        f"{cite('schulman2017ppo')}. PPO uses two neural networks: an Actor that chooses teaching actions and a "
        "Critic that predicts how well the student will perform in the future from state s_t. To avoid sudden "
        "training crashes caused by noisy updates, PPO clips how much the policy is allowed to change in a single "
        f"update step. It typically pairs with Generalized Advantage Estimation (GAE) {cite('schulman2016gae')}, "
        "which blends short-term and long-term reward differences to judge whether an action turned out better or "
        "worse than the Critic expected.",
    )

    add_p(
        doc,
        "Because tutoring has strict pedagogical rules—for example, a tutor cannot ask a harder problem when "
        "already at the hardest level, and should not spam passive lectures without checking understanding—action "
        "masking sets the probability of invalid actions to zero before sampling. Prior studies show the promise of "
        f"RL in education: Ruan et al. (2024) {cite('ruan2024')} found an RL math tutor gave its strongest gains to "
        f"lower-performing students; Nam et al. (2025) {cite('nam2024')} used RL to optimize multi-turn dialogue "
        f"outcomes with an LLM tutor; and CoLA {cite('cola')} introduced latent-action control over LLM responses.",
    )

    add_p(
        doc,
        "A critical pitfall in RL tutoring is optimizing immediate next-question accuracy (a myopic reward). "
        "A myopic RL policy learns a cheap trick: feed easy questions or overly generous hints so the student "
        "gets the next question right immediately. That creates an \"illusion of competence\"—high practice scores "
        "today paired with weak Permastore consolidation and poor 30-day unaided retention. Designing the RL state "
        "and reward explicitly around 1-, 7-, and 30-day delayed retention solves this failure mode.",
    )

    # ------------------------------------------------------------------
    # 2.4 Grounded LLM Tutoring, Hybrid RAG, and Anti-Leakage Guardrails
    # ------------------------------------------------------------------
    add_h2(doc, "2.4 Grounded LLM Tutoring, Hybrid RAG, and Anti-Leakage Guardrails")

    add_p(
        doc,
        "Large Language Models (LLMs) excel at writing clear explanations and fresh coding tasks, but letting an "
        "unconstrained LLM run a classroom introduces serious risks: it may drift off-syllabus, invent wrong test "
        "cases, generate \"easy\" tasks that are actually hard, or accidentally hand the student the exact answer "
        "to an active quiz.",
    )

    add_p(
        doc,
        f"One way to customize an LLM is Low-Rank Adaptation (LoRA) {cite('hu2022lora')}, which trains small "
        "add-on weight matrices while keeping the main model frozen. Our early Stage 1 exploration tested LoRA to "
        "adjust tutor tone (beginner vs. advanced style), showing that shorter technical replies alone do not equal "
        "good teaching or factual accuracy. A stronger alternative for grounding content is Retrieval-Augmented "
        f"Generation (RAG) {cite('lewis2020rag')}, which fetches verified course notes and example problems before "
        "the LLM writes a reply.",
    )

    add_p(
        doc,
        "For programming education, Hybrid RAG combines two complementary search methods. First, BM25 "
        f"{cite('robertson2009bm25')} is a fast keyword-ranking algorithm that matches exact Python tokens like "
        "range() or return statements. Second, TF-IDF (Term Frequency-Inverse Document Frequency) cosine similarity "
        "measures how closely two texts share weighted phrasing across single words and two-word pairs. Merging "
        f"those two ranked lists with Reciprocal Rank Fusion (RRF) {cite('cormack2009rrf')} (using the standard "
        "smoothing constant k = 60 so top ranks stay balanced) rewards documents ranked "
        "near the top by either method without needing fragile score scaling.",
    )

    add_p(
        doc,
        f"Recent work has also explored aligning tutor dialogue and grading with student learning. Scarlatos et al. "
        f"(2025) {cite('scarlatos2024')} scored candidate LLM replies using predicted student correctness and a "
        "teaching rubric. Meanwhile, structured evaluation frameworks such as TypeSafe's Score and Confidence "
        f"primitives {cite('typesafe1')}, {cite('typesafe2')} highlight an important rule for grading free-text "
        "explanations: the rubric score (did the student state the key concept?) and the grader's confidence in "
        "that score are separate quantities and must not be mixed into one vague number. For Python code exercises, "
        "deterministic Abstract Syntax Tree (AST) sandboxing executes student code safely against fixed unit tests "
        "rather than relying on LLM guesswork, while anti-leakage guardrails scan tutor replies so students cannot "
        "fish for or paste active quiz questions to extract answers.",
    )

    # ------------------------------------------------------------------
    # 2.5 Research Gaps Addressed by This Thesis
    # ------------------------------------------------------------------
    add_h2(doc, "2.5 Research Gaps Addressed by This Thesis")

    add_p(
        doc,
        "Bringing these four lines of research together reveals clear gaps that existing systems leave unsolved. "
        "First, deep knowledge tracing models rarely enforce cognitive invariants—such as non-increasing forgetting "
        "over breaks, Bahrick Permastore floors, and immunity to rapid guessing—while also exposing a compact "
        "long-term memory state. Second, RL tutors often optimize immediate accuracy or rely on unconstrained "
        "prompting rather than gated actions tied to 30-day retention. Third, LLM tutors frequently lack hybrid "
        "curriculum grounding, difficulty-locked task repair, AST sandbox grading, and active-quiz anti-leakage "
        "guardrails. Table 2.1 summarizes how our unified architecture bridges these gaps.",
    )

    headers = [
        "Research Dimension",
        "Standard KT (BKT / DKT / SAKT)",
        "Spaced Schedulers (HLR / FSRS)",
        "Prior RL & LLM Tutors",
        "Unified Architecture (This Thesis)",
    ]
    rows = [
        [
            "Memory Horizon & Forgetting",
            "Short next-step accuracy; no Permastore floor or multi-week power-law tail.",
            "Power-law / half-life decay, but no neural context or multi-format 3PL IRT.",
            "Often stateless chat history or short-horizon mastery heuristics.",
            "Monotonic Grey-Box GRU: HLR half-life + power-law decay + Permastore floor b_t + 3PL IRT.",
        ],
        [
            "Behavioral Sanity & Rapid Guessing",
            "Unconstrained RNNs/Transformers can boost skill on 6s wrong guesses or drift up on breaks.",
            "Assumes honest self-ratings or flashcard grades without hint/timing confounding.",
            "Hint leakage and rapid guessing rarely separated in state.",
            "6-check behavioral gate: timing masked on failures, strictly non-increasing forgetting, O(1) state.",
        ],
        [
            "Closed-Loop RL Control",
            "Passive predictor only; does not choose teaching actions.",
            "Schedules review intervals only; no hint, example, or self-explanation policy.",
            "Myopic rewards or soft LLM suggestions that can drift or spam lectures.",
            "8-action Gated Option B PPO maximizing 1/7/30-day retention with hard valid-action masks.",
        ],
        [
            "Grounded LLM & Guardrails",
            "Not applicable (prediction only).",
            "Static flashcards without dialogue or code grading.",
            "Style LoRA or unguarded prompts risk off-curriculum drift and answer leakage.",
            "Hybrid RAG (FTS5 BM25 + TF-IDF + RRF k=60), AST sandbox, difficulty-locked Q-gen, EduGuard.",
        ],
    ]
    add_table(
        doc,
        headers,
        rows,
        caption="Table 2.1 — Comparison of Prior Tutoring Literature and This Thesis Across Four Research Gaps.",
        col_widths=[1.40, 1.35, 1.35, 1.35, 1.35],
    )
