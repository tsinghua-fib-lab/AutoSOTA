<div align="center">

<img src="./assets/readme-hero.svg" width="100%" alt="AutoSOTA — Automated optimization of research codebases" />

<br />
<br />

Developed by **FIB LAB, Tsinghua University** and **Zhongguancun Academy**

[![CLI v0.3.1](https://img.shields.io/badge/CLI-v0.3.1-9cffb3?style=flat-square&labelColor=101820)](./cli_guide.md) [![Research paper](https://img.shields.io/badge/Paper-arXiv%3A2604.05550-ffdb94?style=flat-square&labelColor=101820&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.05550) [![Explore results](https://img.shields.io/badge/Experiments-Explore-a8c7fa?style=flat-square&labelColor=101820&logo=github&logoColor=white)](#experiment-results)

**[Project website](https://tsinghua-fib-lab.github.io/AutoSOTA/)** &nbsp;·&nbsp; **[Experiment results](#experiment-results)** &nbsp;·&nbsp; **[CLI guide](./cli_guide.md)** &nbsp;·&nbsp; **[Citation](#citation)**

</div>

<br />

AutoSOTA automatically explores research codebases, proposes improvements, runs experiments, and evaluates the results. This repository brings together our experiment archives and the guide to running AutoSOTA locally.

<table>
<tr>
<td width="50%" valign="top">
<h3>🧪 Explore the research</h3>
<p>Conference experiments and medical benchmark studies, with source code, reports, and evaluation protocols.</p>
<a href="#experiment-results"><strong>Browse experiment results →</strong></a>
</td>
<td width="50%" valign="top">
<h3>🛠️ Run AutoSOTA locally</h3>
<p>Start from a runnable research repository and an optimization goal. Let the CLI drive the research loop.</p>
<a href="./cli_guide.md"><strong>Open the CLI guide →</strong></a>
</td>
</tr>
</table>

---

<a id="experiment-results"></a>

## 🧪 Experiment Results

### Conference Experiments

| Batch | Research archive | Explore |
|:---:|:---|:---|
| **01** | **Early experiments** · 105 papers across multiple conferences | [Code & reports](./improved-papers/) · [Results](https://tsinghua-fib-lab.github.io/AutoSOTA/#papers) |
| **02** | **CVPR 2026** | [Code & reports](./CVPR-2026/) · [Dashboard](https://tsinghua-fib-lab.github.io/AutoSOTA/cvpr.html) |
| **03** | **ICML 2026** | [Code & reports](./ICML-2026/) · [Dashboard](https://tsinghua-fib-lab.github.io/AutoSOTA/icml.html) |

### Medical Benchmarks

**ODELIA Breast MRI Challenge 2025**

Healthy / benign / malignant breast-ROI classification with independent heads for seven frozen encoders.

| Released baseline | AutoSOTA | Folds with improved Challenge score |
|:---:|:---:|:---:|
| 0.876448 | **0.899644** | **7 / 7** |

*Challenge score on the same 300 public-test breast ROIs.*

[Read the research report →](./medical-benchmarks/ODELIA-AutoSOTA/docs/research_report.md) &nbsp;·&nbsp; [Code & results](./medical-benchmarks/ODELIA-AutoSOTA/) &nbsp;·&nbsp; [Complete search history](./medical-benchmarks/ODELIA-AutoSOTA/docs/search_history.md)

> [!NOTE]
> Scores compare the same public-test cohort. Public-test feedback informed the search, and released encoder training may overlap this cohort. This is not a new official hidden-test leaderboard result.

### Experiment News

| Date | Research milestone |
|:---|:---|
| **2026-10-10** | Published the **ODELIA medical benchmark study**, with final code, aggregate results, and an English Markdown report and search history. |
| **2026-07-06** | Added the **ICML 2026** experiment batch, with paper-level reproduction and optimization tracking. |
| **2026-06-07** | Published the **CVPR 2026** experiment batch. |
| **2026-03-27** | Published our **early experiments** across multiple conferences. |

---

## 🛠️ AutoSOTA CLI

Bring a runnable codebase and an optimization goal. **AutoSOTA CLI** iteratively researches strategies, modifies code, runs experiments, and reports the results.

| You provide | AutoSOTA runs | You receive |
|:---|:---|:---|
| Research codebase & optimization goal | Research → code → experiment → evaluate | Optimized code, patches & experiment reports |

**[Installation, configuration & usage →](./cli_guide.md)**

### CLI Updates

**v0.3.1 · July 27, 2026** — Cleaner session output: one session-start message, with retry and error notices kept visible.

<details>
<summary><strong>View the full CLI release history</strong></summary>

- **2026-07-27** 🔧 Released **v0.3.1** — Fixed log spam: some backends (third-party relays in particular) emit status events continuously, repeating the `[session]` line until it buried the useful output. It is now printed once per session, keeping only retry and error notices.

- **2026-06-28** 🚀 Released **v0.3.0** — Ported the multi-agent optimization "brain" into the local CLI: resumable runs (`autosota --resume`); a continuation supervisor that fixes premature aborts by relaunching until the budget is spent; an independent read-only evaluation pass (`evaluation_verdict.json`); removal of percentage-target early stopping in favor of a budget-only, structured multi-metric objective with guardrails; deep-research web-search verification (`web_search VERIFIED / UNVERIFIED`); and a fix for workspace cross-talk.

- **2026-05-15** 🔧 Released **v0.2.1** — fixes CLI invocation failure on clean machines after global install; adds `autosota login` for one-time terms-of-service acceptance.

- **2026-05-13** 📝 Added detailed configuration instructions for both DeepSeek official direct connection and calling DeepSeek via OpenRouter.

- **2026-05-12** 🚀 Released v0.2.0 — Supported direct access to domestic models like DeepSeek (no longer tied to OpenRouter), allowed the use of separate API keys for code optimization and literature review, and added anti-tampering protection for evaluation scripts.

- **2026-04-29** 🔧 Released v0.1.1 — Added inter-round pause functionality (-i), allowing you to check progress or add instructions without killing the process; introduced new interactive commands including ask, steer, and sessions.

- **2026-04-22** 🎉 Released v0.1.0 — Initial release, achieving a closed loop for automated research code optimization: automatically explores repository configurations, reviews relevant literature, iteratively modifies code, and exports the optimal code and patch upon completion.

- **2026-04-21** 💻 Initialized the AutoSOTA CLI project on the laboratory server.

</details>

---

## ⭐ Star History

<details>
<summary><strong>See how the community is growing</strong></summary>

[![Star History Chart](https://api.star-history.com/svg?repos=tsinghua-fib-lab/AutoSOTA&type=Date)](https://star-history.com/#tsinghua-fib-lab/AutoSOTA&Date)

</details>

---

<a id="citation"></a>

## 📝 Citation

If you find AutoSOTA useful for your research, please cite:

```bibtex
@article{li2026autosota,
  title={Autosota: An end-to-end automated research system for state-of-the-art ai model discovery},
  author={Li, Yu and Shao, Chenyang and Liu, Xinyang and Zhao, Ruotong and Liu, Peijie and Su, Hongyuan and Chen, Zhibin and Yang, Qinglong and Xu, Anjie and Fang, Yi and others},
  journal={arXiv preprint arXiv:2604.05550},
  year={2026}
}
```
