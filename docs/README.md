<div align="center">

# 🚀 AutoSota

**Automated optimization of research codebases**

Developed by **FIB LAB, Tsinghua University** and **Zhongguancun Academy**

[![AutoSota](https://img.shields.io/badge/Generated%20by-AutoSota-6f42c1?style=for-the-badge&logo=github&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Active-0a7?style=for-the-badge&logo=activity&logoColor=white)](#)

</div>

<br>

> **Overview** — AutoSota automatically optimizes research codebases. This repository
> provides the CLI usage guide and the code and report archives from three experiment
> batches: early experiments across multiple conferences, CVPR 2026, and ICML 2026.

---

## 🧪 Experiment Results

| Batch | Scope | Entry points |
|:---:|:---|:---|
| 1 | Early experiments across multiple conferences (105 papers) | [Code and reports](../improved-papers/) · [Results](https://tsinghua-fib-lab.github.io/AutoSOTA/#papers) |
| 2 | CVPR 2026 | [Code and reports](../CVPR-2026/) · [Results dashboard](https://tsinghua-fib-lab.github.io/AutoSOTA/cvpr.html) |
| 3 | ICML 2026 | [Code and reports](../ICML-2026/) · [Results dashboard](https://tsinghua-fib-lab.github.io/AutoSOTA/icml.html) |

### Experiment News

- **2026-07-06** 🧪 Added the **third experiment batch: ICML 2026**, with paper-level reproduction and optimization tracking.

- **2026-06-07** 🧪 Published the **second experiment batch: CVPR 2026**.

- **2026-03-27** 🧪 Published the **first experiment batch: 105 papers across multiple conferences**, representing our early experiments.

---

## 🛠️ AutoSota CLI

We also provide **AutoSota CLI**, a command-line tool for automatically optimizing research codebases.

With AutoSota CLI, you can provide a runnable codebase and an optimization goal, and the tool will iteratively propose strategies, modify code, run experiments, and report improved results.

For installation and usage instructions, please see [cli_guide.md](../cli_guide.md).

### CLI Updates

- **2026-07-27** 🔧 Released **v0.3.1** — Fixed log spam: some backends (third-party relays in particular) emit status events continuously, repeating the `[session]` line until it buried the useful output. It is now printed once per session, keeping only retry and error notices.

- **2026-06-28** 🚀 Released **v0.3.0** — Ported the multi-agent optimization "brain" into the local CLI: resumable runs (`autosota --resume`); a continuation supervisor that fixes premature aborts by relaunching until the budget is spent; an independent read-only evaluation pass (`evaluation_verdict.json`); removal of percentage-target early stopping in favor of a budget-only, structured multi-metric objective with guardrails; deep-research web-search verification (`web_search VERIFIED / UNVERIFIED`); and a fix for workspace cross-talk.

- **2026-05-15** 🔧 Released **v0.2.1** — fixes CLI invocation failure on clean machines after global install; adds `autosota login` for one-time terms-of-service acceptance.

- **2026-05-13** 📝 Added detailed configuration instructions for both DeepSeek official direct connection and calling DeepSeek via OpenRouter.

- **2026-05-12** 🚀 Released v0.2.0 — Supported direct access to domestic models like DeepSeek (no longer tied to OpenRouter), allowed the use of separate API keys for code optimization and literature review, and added anti-tampering protection for evaluation scripts.

- **2026-04-29** 🔧 Released v0.1.1 — Added inter-round pause functionality (-i), allowing you to check progress or add instructions without killing the process; introduced new interactive commands including ask, steer, and sessions.

- **2026-04-22** 🎉 Released v0.1.0 — Initial release, achieving a closed loop for automated research code optimization: automatically explores repository configurations, reviews relevant literature, iteratively modifies code, and exports the optimal code and patch upon completion.

- **2026-04-21** 💻 Initialized the AutoSOTA CLI project on the laboratory server.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tsinghua-fib-lab/AutoSOTA&type=Date)](https://star-history.com/#tsinghua-fib-lab/AutoSOTA&Date)

---

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
