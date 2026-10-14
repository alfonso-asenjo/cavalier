# Related work: LLM advisors, deliberation and crisis decision-making (for Cavalier advisor agents)

Research date: 2026-10-14. "Preprint" = arXiv-only, not peer reviewed. Items marked [bib-only] were not fetched this session; only bibliographic identity is asserted, not findings.

## (a) LLM wargame / foreign-policy / escalation simulations

### Takeaway
Every major LLM wargame study finds escalation bias and limited de-escalation. The most relevant result for advisors is Lamparth et al.: when an LLM simulates a team's internal discussion, the dialogue is harmonious and shallow, and even extreme persona traits ("pacifist" vs "aggressive sociopath") made no significant difference. That is the null result Cavalier's advisors must beat.

### Cited Findings
- **Rivera, Mukobi, Reuel, Lamparth, Smith, Schneider (FAccT 2024; arXiv Jan 2024)**, "Escalation Risks from Language Models in Military and Diplomatic Decision-Making". Multi-agent nation wargame with a scoring framework. All five off-the-shelf LLMs showed "forms of escalation and difficult-to-predict escalation patterns", including arms-race dynamics, and in rare cases nuclear use with deterrence/first-strike justifications. Paper: 10 pages plus a 57-page appendix with the action list and nation design. — [arXiv 2401.03408](https://arxiv.org/abs/2401.03408)
- Secondary summary of Rivera et al.: none of the models de-escalated over a simulation. — [Euronews coverage](https://www.euronews.com/2024/02/22/ai-models-chose-violence-and-escalated-to-nuclear-strikes-in-simulated-wargames) (secondary)
- **Lamparth, Corso, Ganz, Mastro, Schneider, Trinkunas (preprint, Mar 2024, v4 Oct 2024)**, "Human vs. Machine". 214 national-security experts in a fictional US–China crisis wargame, compared with LLM-simulated teams. Two LLM modes: decide directly, or first simulate a dialogue among team members. Findings:
  - High-level agreement with humans, but significant differences in individual actions. LLMs were sometimes more aggressive and highly sensitive to scenario changes.
  - Simulated team dialogue lacked quality and kept a "farcical harmony".
  - No significant difference even for extreme player traits ("pacifist", "aggressive sociopath").
  - Outcomes depended on the model and on direct-vs-dialogue mode.
  — [arXiv 2403.03407](https://arxiv.org/abs/2403.03407)
- **WarAgent (Hua, Fan, Li, Mei, Ji, Ge, Hemphill, Zhang; preprint Nov 2023, v2 Jan 2024)**. LLM multi-agent system simulating the countries in WWI, WWII and China's Warring States period, used to study the triggers and conditions of war. The abstract gives no quantitative results. — [arXiv 2311.17227](https://arxiv.org/abs/2311.17227)
- **Snow Globe (Hogan & Brennen, IQT Labs; preprint Apr 2024)**, "Open-Ended Wargames with Large Language Models".
  - Distinguishes quantitative (discrete-move) from qualitative (open-ended) wargames and argues LLMs make it possible to automate the qualitative kind.
  - Open-source implementation, tested on an AI incident-response tabletop exercise and a geopolitical crisis game.
  - Each stage can be under AI or human control.
  — [arXiv 2404.11446](https://arxiv.org/abs/2404.11446); [PyPI llm-snowglobe](https://pypi.org/project/llm-snowglobe/0.1.0)
- CIA Directorate of Digital Innovation and In-Q-Tel used Snow Globe for human–AI teaming in war games, with humans playing alongside or against simulated personas. Studies in Intelligence 69(4), Dec 2025; only the extract was seen. — [CIA CSI](https://www.cia.gov/resources/csi/studies-in-intelligence/studies-in-intelligence-vol-69-no-4-extracts-december-2025/snow-globe-multi-player-ai-system-lessons-from-human-ai-teaming-in-war-games)
- **Payne (King's College London; preprint, 16 Feb 2026)**, "AI Arms and Influence". GPT-5.2, Claude Sonnet 4 and Gemini 3 Flash played opposing leaders in nuclear crises. Findings:
  - Models attempted deception (signals they did not intend to follow) and showed theory-of-mind reasoning.
  - Threats more often provoked counter-escalation than compliance, and high mutual credibility accelerated conflict.
  - "No model chose accommodation or withdrawal". Models only reduced their level of violence.
  — [arXiv 2602.14740](https://arxiv.org/abs/2602.14740)
- Payne study, as reported by KCL and press:
  - 21 games, with nuclear signalling in about 95% of games.
  - An injected "fog of war" accident mechanism could only push choices toward escalation.
  - RAND's Edward Geist argued the victory conditions made nuclear war look winnable.
  - Two of the three full nuclear wars were reportedly triggered by the accident mechanism.
  — [KCL news](https://www.kcl.ac.uk/news/artificial-intelligence-under-nuclear-pressure-first-large-scale-kings-study-reveals-how-ai-models-reason-and-escalate-under-crisis); [TechXplore](https://techxplore.com/news/2026-03-ai-escalates-nuclear-action-war.html) (secondary; the exact metrics are inconsistent across outlets)
- **Elbaum & Panter (preprint, Aug 2025)**, "Managing Escalation in Off-the-Shelf LLMs". Two simple, non-technical prompt interventions inserted into an earlier study's wargame "substantially reduced escalation". The specific techniques were not visible in the abstract. — [arXiv 2508.01056](https://arxiv.org/abs/2508.01056)
- 2026 preprint "Don't Want Your LLM to Recommend Nuclear Strike? Try Asking It in Japanese": prompt language may shift escalation recommendations. Title only; not read. — [arXiv 2608.12373](https://arxiv.org/pdf/2608.12373)
- **GovSim (Piatti et al., NeurIPS 2024)**, "Cooperate or Collapse". In this commons simulation:
  - All but the most powerful LLMs failed to sustain the resource; the best survival rate was below 54%.
  - Communication reduced overuse by 21%.
  - Failures were attributed to an inability to reason about the long-term effects of actions on the group equilibrium.
  - Prompting "Universalization" moral reasoning helped.
  — [arXiv 2404.16698](https://arxiv.org/abs/2404.16698); [NeurIPS page](https://neurips.cc/virtual/2024/99392)
- **Diplomacy**:
  - "Democratizing Diplomacy" (preprint Aug 2025) is a harness for evaluating any LLM on full-press Diplomacy. It measures betrayal as the share of promises later broken; rates vary by model. — [arXiv 2508.07485](https://arxiv.org/pdf/2508.07485)
  - The 2025 "AI Diplomacy" open-source project pitted frontier models against each other. Press coverage claims distinct model "personalities", e.g. Claude reluctant to deceive; this is anecdotal. — [Every.to](https://every.to/p/diplomacy) (blog)
  - CICERO (Meta FAIR, Science 2022) is the canonical human-level Diplomacy agent, combining a language model with strategic planning. [bib-only] — [Science doi:10.1126/science.ade9097](https://doi.org/10.1126/science.ade9097)

### Inferences
- With a single leader agent, Cavalier will probably reproduce the escalation bias and the missing withdrawal/accommodation options (Rivera, Payne). Advisors are a plausible lever: an advisor whose explicit role is to propose de-escalation or exit options can put options on the table that the leader would otherwise never generate.
- Lamparth's "farcical harmony" and trait-insensitivity results predict the main risk: 4–6 qwen3.5:9b personas will sound alike unless their differences come from structure (different information, roles and objectives), not just name and biography.
- Payne's critique shows that game mechanics such as victory conditions and accident injection can dominate outcomes. Cavalier should keep game mechanics fixed when comparing "with vs. without advisors".
- Elbaum & Panter suggest that even light prompt framing moves escalation a lot. This is a validity threat (advisor prompts act as escalation interventions) but also a design tool.

### Gaps
- Rivera et al. full agent design (8 nations, ~27 actions, turn structure) and base-vs-RLHF findings were not verified in this session; check the appendix.
- WarAgent's internal architecture, including any "secretary" verification agent, was not confirmed from the abstract.
- Elbaum & Panter's two interventions are unknown; read the full text.
- No study found uses historical *advisor* personas (e.g. ExComm) around a leader agent; this looks like an open niche.

## (b) Multi-agent deliberation designs (debate, councils, advisors)

### Takeaway
Debate helps on tasks with a right answer when agents start out disagreeing a moderate amount and the number of rounds is limited. It collapses toward consensus or sycophancy as rounds, connectivity and authority cues increase. For open-ended policy choices, independence before exposure and model heterogeneity are what preserve diversity.

### Cited Findings
- **Du, Li, Torralba, Tenenbaum, Mordatch (ICML 2024; arXiv May 2023)**. Multiple instances answer, then debate over rounds. They "almost always converge" on a single, more accurate answer, with fewer hallucinated facts. Convergence is the *goal* here, unlike in Cavalier. — [arXiv 2305.14325](https://arxiv.org/abs/2305.14325); [ICML 2024](https://icml.cc/virtual/2024/poster/32620)
- **Liang et al. (EMNLP 2024; arXiv May 2023)**, "Encouraging Divergent Thinking … Multi-Agent Debate" (MAD). Findings:
  - Defines "Degeneration-of-Thought": once confident, a self-reflecting LLM can no longer generate novel thoughts.
  - In MAD, debaters argue "tit for tat" and a judge decides.
  - The debate needs an adaptive break (stop early), and performance requires a *modest* level of confrontation, not extreme.
  - LLM judges may be unfair when the debaters are different models.
  — [arXiv 2305.19118](https://arxiv.org/abs/2305.19118)
- **ChatEval (Chan et al., ICLR 2024)**. A multi-agent referee team for evaluation; it is commonly cited for the finding that diverse role prompts matter. [bib-only] — [arXiv 2308.07201](https://arxiv.org/abs/2308.07201)
- **"Talk Isn't Always Cheap" (Wynn, Satija, Hadfield; ICML 2025 MAS workshop)**. Findings:
  - Debate can *lower* accuracy, because models often flip from correct to incorrect answers after seeing peer reasoning.
  - Paying for correctness did not reduce these flips.
  - A weaker agent disrupts stronger ones, and longer debates can degrade further.
  — [arXiv 2509.05396](https://arxiv.org/abs/2509.05396)
- **"Peacemaker or Troublemaker" (Yao et al., preprint Sep 2025)**. Sycophancy is a core failure mode in multi-agent debate; it causes premature collapse of disagreement and can fall below single-agent baselines. — [arXiv 2509.23055](https://arxiv.org/abs/2509.23055v1)
- **Chen et al. (ACL 2026 Findings; arXiv Apr 2026)**, "Diversity Collapse in Multi-Agent LLM Systems". Findings:
  - Stronger, more aligned models show diminishing marginal diversity.
  - Authority-driven dynamics suppress semantic diversity compared with junior-led groups.
  - More agents give diminishing returns, and dense communication topologies accelerate premature convergence.
  - The authors attribute collapse mainly to interaction structure, not to the model.
  - Code: github.com/Xtra-Computing/MAS_Diversity.
  — [arXiv 2604.18005](https://arxiv.org/abs/2604.18005)
- **Sela (preprint, Apr 2026)**, "Preserving Disagreement: Architectural Heterogeneity and Coherence Validation in Multi-Agent Policy Simulation". Setup and results:
  - An "AI Council" of value-perspective evaluators ran 120 deliberations on child-welfare and housing scenarios.
  - Giving each perspective a *different* 7–9B model cut first-choice concentration from 70.9% to 46.1% (child welfare) and from 46.0% to 22.9% (housing), both p<0.001, versus the same model for all.
  - Three Delphi designs failed.
  - 8B models responded to counter-arguments in a binary, not graded, way.
  - A frontier-model "coherence validation" judge, which checks that reasoning is grounded in the assigned values, showed a fidelity–diversity tradeoff.
  — [arXiv 2604.26561](https://arxiv.org/abs/2604.26561)

### Inferences
- Do not use multi-round advisor debate on a 9B model. Use a single round of *independent* advice: each advisor sees the state plus its private briefing, not the other advisors' advice. The leader sees all of it. This avoids the conformity cascade (Wynn; Chen) and also caps cost.
- Avoid "authority cues" in advisor prompts, e.g. telling advisors what the leader currently leans toward, since authority suppresses diversity (Chen). Give the leader's prior intention to advisors only if leader-sycophancy is the very thing being tested.
- Liang's "modest confrontation" result suggests advisors should argue clearly different positions, but not be forced to contradict everything.
- Sela is the closest analogue to Cavalier: a policy council with small 7–9B models and no correct answer. It is a preprint, but it directly supports trying 2–3 different small models for advisors if VRAM and latency allow (e.g. qwen3.5:9b plus a llama/gemma/mistral 7–9B, swapped sequentially in Ollama).

### Gaps
- No controlled study found of an "advisors → single decider" topology specifically, as opposed to peer debate or voting. The closest are Lamparth's dialogue mode and Sela's council.
- The exact mechanics of ChatEval role diversity and of Liang's affirmative/negative role setup were not verified this session.

## (c) Failure mode: personas from the same model converge (homogeneity, sycophancy, conformity, drift)

### Takeaway
There is strong evidence that same-model personas share one "mode". Persona descriptors explain little variance, open-ended outputs collapse across and even within models, persona adherence decays within a few turns, and agents conform to majorities or peers. Persona labels alone will not produce diverse advisors.

### Cited Findings
- **Artificial Hivemind (Jiang et al., NeurIPS 2025 Best Paper; arXiv Oct 2025)**. Built Infinity-Chat (26K open-ended queries, 31,250 human annotations) and found pronounced intra-model repetition *and* inter-model homogeneity in open-ended generation. LM judges and reward models are poorly calibrated on items where humans disagree. — [arXiv 2510.22954](https://arxiv.org/abs/2510.22954); [NeurIPS](https://neurips.cc/virtual/2025/poster/121421)
- **Hu & Collier (ACL 2024)**, "Quantifying the Persona Effect in LLM Simulations". Persona variables explain under 10% of annotation variance in subjective datasets. Persona prompting gives small but significant gains, mostly where disagreement is real but minor. — [ACL Anthology 2024.acl-long.554](https://aclanthology.org/2024.acl-long.554); [arXiv 2402.10811](https://arxiv.org/abs/2402.10811v2)
- **Wang, Morgenstern, Dickerson (arXiv Feb 2024; journal version claimed but not verified)**. Studied 3,200 participants, 16 identities and 4 LLMs. LLMs misportray and *flatten* groups, and identity prompts can essentialize. Inference-time mitigations reduce but do not remove these harms. — [arXiv 2402.01908](https://arxiv.org/abs/2402.01908)
- Follow-up work (arXiv 2511.21218) reports that "identity flattening" homogeneity persists after persona assignment and RAG mitigations. — [arXiv 2511.21218](https://arxiv.org/pdf/2511.21218v2)
- **Argyle et al. (Political Analysis 2023)**, "Out of One, Many". This is the original "silicon sampling" proposal and the baseline the critiques above respond to. [bib-only] — [arXiv 2209.06899](https://arxiv.org/abs/2209.06899)
- **Persona drift: Li, Liu, Bashkansky, Bau, Viégas, Pfister, Wattenberg (COLM 2024)**. In self-chats between instructed bots, there was significant drift away from system-prompt personas within 8 rounds for LLaMA2-chat-70B and GPT-3.5, attributed to attention decay over long context. They propose split-softmax, a training-free fix. Code: github.com/likenneth/persona_drift. — [arXiv 2402.10962](https://arxiv.org/abs/2402.10962)
- **Homogeneous-debate conformity in 7–8B models (2026 preprints, abstracts via search)**:
  - Sycophantic adoption of the majority answer up to 85.5%.
  - Plurality voting discards correct answers present in the pool (oracle gap up to 32.3 pp).
  - More than 1 in 10 initially correct majorities are lost in 3-agent homogeneous debate for the most susceptible models.
  - Separately, convergence scores above 0.88 were reported between *opposing* personas in a social-simulation context; the source was not identified precisely.
  — [arXiv 2605.00914](https://arxiv.org/abs/2605.00914); [arXiv 2609.35279](https://arxiv.org/html/2609.35279); search-summary claims, not fetched
- A 2026 preprint finds that mixed-model populations reach weaker biased consensus than homogeneous ones. — [arXiv 2608.02827](https://arxiv.org/html/2608.02827v1) (title only)
- Lamparth et al.: no significant behavioral difference even for extreme persona traits. — [arXiv 2403.03407](https://arxiv.org/abs/2403.03407)

### Inferences
- For Cavalier there are three distinct risks, each needing its own check:
  1. **Inter-advisor homogeneity**: all advisors say the same thing.
  2. **Leader echo or sycophancy**: advisors mirror the leader, or the leader rubber-stamps.
  3. **Drift**: personas lose their stance over a long game. Risk rises with context length; Li et al. saw drift within 8 rounds.
- Drift mitigation: rebuild each advisor's prompt from scratch every key turn (persona card plus compact state summary) instead of appending to a growing chat history. This keeps the persona at the start of a short context.
- Smaller, more aligned models tend toward a shared "helpful moderate" mode (Chen; Jiang), so the default failure is probably every advisor recommending "cautious diplomacy plus quarantine".

### Gaps
- No study quantifying persona homogeneity specifically for qwen3.5-class 9B models was found.
- The "0.88 convergence between opposing personas" figure could not be traced to a specific paper.

## (d) Mitigation and validation techniques (make advisors diverse, and verify it)

### Takeaway
Diversity has to be built into the structure: different information, explicit stance anchors, independent generation, optionally different models. It then has to be measured with action-level, not just text-level, metrics against a "leader alone" ablation.

### Cited Findings
- **Model heterogeneity** reduced concentration on the first-choice option by about 23–25 pp in a small-model policy council. — [Sela 2026, arXiv 2604.26561](https://arxiv.org/abs/2604.26561)
- **Independence and sparse topology**: dense communication accelerates premature convergence, and the authors recommend preserving independence and disagreement. — [Chen et al. 2026, arXiv 2604.18005](https://arxiv.org/abs/2604.18005)
- **Limit rounds, with moderate confrontation**: use an adaptive break and a modest level of tit-for-tat. — [Liang et al., arXiv 2305.19118](https://arxiv.org/abs/2305.19118)
- **Incentives alone don't stop sycophantic flips**. — [Wynn et al., arXiv 2509.05396](https://arxiv.org/abs/2509.05396)
- **Coherence/fidelity judge**: a frontier judge that checks reasoning is grounded in the assigned values. Useful for fidelity, but it can *increase* convergence when options are competitive. — [Sela 2026](https://arxiv.org/abs/2604.26561)
- **Persona-stability control**: split-softmax, or short contexts, to fight attention decay. — [Li et al. COLM 2024](https://arxiv.org/abs/2402.10962)
- **Persona richness has limited payoff**: fine-grained persona detail added little lexical diversity beyond simple constraints such as a length cutoff. — [Kambhatla et al., Findings EMNLP 2025](https://preview.aclanthology.org/dashboard/2025.findings-emnlp.1146)
- **Validation metrics used in the literature**:
  - First-choice concentration across evaluators. — [Sela](https://arxiv.org/abs/2604.26561)
  - Semantic diversity of outputs. — [Chen](https://arxiv.org/abs/2604.18005)
  - Correct→incorrect flip counts. — [Wynn](https://arxiv.org/abs/2509.05396)
  - Modal adoption rate and oracle gap. — [arXiv 2605.00914](https://arxiv.org/abs/2605.00914)
  - Drift measured by persona-adherence probes over rounds. — [Li](https://arxiv.org/abs/2402.10962)
  - Variance explained by persona variables. — [Hu & Collier](https://aclanthology.org/2024.acl-long.554)
  - Promise-vs-action betrayal rate. — [arXiv 2508.07485](https://arxiv.org/pdf/2508.07485)
  - Human-vs-LLM action distributions. — [Lamparth](https://arxiv.org/abs/2403.03407)
  - Escalation scoring over time. — [Rivera](https://arxiv.org/abs/2401.03408)

### Inferences (design recommendations for Cavalier; not empirically validated for this setup)
1. **Structure over biography.** Give each advisor:
   - a *private briefing* with different facts or emphases: e.g. the JCS (LeMay/Taylor) gets the military-readiness and strike-option estimate; McNamara gets the blockade options and force posture; Stevenson/Ball get UN and allied reactions; RFK gets the moral "Pearl Harbor in reverse" framing and back-channel status.
   - an *explicit stance anchor*: a preferred action class from the game's action enum and a red line.
   - *required output fields*: recommended action (must be from the enum), confidence, a one-sentence risk, and what would change its mind.
2. **Independent-first.** Generate all advisor outputs from the same state without exposure to each other or to the leader's tentative choice. Add a second exposure round only if testing it, and log whether advisors switch.
3. **Assign a dissent role.** Make one advisor a rotating devil's advocate, required to argue for the strongest alternative to the current modal recommendation. Ask a "withdrawal/accommodation" advocate to state an exit option explicitly; Payne found models never chose these unprompted.
4. **Optional heterogeneity.** Run 2–3 different 7–9B models for advisors, sequentially in Ollama (one loaded at a time on 8 GB). Measure whether this beats single-model diversity, following Sela.
5. **Sampling.** Use temperature or seed variation per advisor, but treat it as noise, not perspective diversity (Jiang: intra-model repetition persists).
6. **Validation suite**:
   - **(i) Same-state probe.** Freeze N game states and run all advisors K times each. Measure:
     - the action-recommendation distribution per advisor (entropy; pairwise Jensen–Shannon divergence);
     - concentration on the first-choice action across advisors (Sela's metric);
     - pairwise embedding distance between advice texts, compared with between-sample distance within the same advisor;
     - stance classification by an LLM judge that blindly identifies "which advisor wrote this". If the judge can attribute advice to the right advisor, personas are distinct.
   - **(ii) Ablations.** Compare leader alone, leader plus advisors with personas as labels only, leader plus advisors with briefings and anchors, and optionally heterogeneous models. Track final outcome, escalation score per turn, and the rate at which the leader's action equals the majority advice.
   - **(iii) Sycophancy test.** Inject the leader's tentative choice into advisor prompts and measure how far recommendations shift toward it.
   - **(iv) Drift test.** Compare stance and anchor adherence at early vs. late key turns.
   - **(v) Trajectory diversity** across seeds/runs: outcome entropy (blockade / strike / invasion / deal / nuclear) and edit distance between action sequences, comparing conditions.
7. **Cost (small local models).** 4–6 advisors × short outputs (≈150–250 tokens) only on key turns adds roughly 4–6 extra generations per key turn. With fresh, short per-turn prompts, context stays bounded. This estimate is inferred; no measured latency figures were found in the literature for 9B local setups.

### Gaps
- No published latency or VRAM numbers were found for multi-persona deliberation on 8 GB consumer GPUs; Cavalier will have to measure its own.
- No validated metric specific to "advisor influence on a single decider" was found; the leader-follows-majority rate and ablation outcome deltas proposed above are the designer's own constructions.
- Whether devil's-advocate roles reliably beat plain independent generation on small models has not been established in sources found this session.
