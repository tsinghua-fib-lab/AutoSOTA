# RealTravel

This repository contains the RealTravel dataset.

RealTravel is a novel dataset created to facilitate research in personalized, preference-driven travel planning. It extends and enhances the existing **TravelPlanner** benchmark by incorporating authentic user reviews and point-of-interest (POI) metadata from the **Google Local** dataset. The dataset is designed to address the challenge of modeling the implicit travel preferences of real-world users, moving beyond a reliance on synthetic user profiles.

## Key Features

* **Real-World User Data:** Incorporates actual user reviews and metadata from Google Local, providing a foundation for developing and evaluating personalized travel planning systems grounded in real user behaviors.
* **Focus on Implicit Preferences:** Enables the study and modeling of users' implicit preferences, which are often not explicitly stated in travel queries.
* **Structured and Symbolic Queries:** Includes a travel request generator that produces structured, symbolic queries from natural language descriptions, ensuring that user requests are machine-readable and can be used for rigorous evaluation.
* **Comprehensive Data:** The dataset consists of 1,000 test samples and 155 validation samples. The underlying database contains information on 77 cities, and tens of thousands of restaurants, attractions, and accommodations.

## Dataset Structure

The RealTravel dataset is structured to support travel planning tasks. Each instance consists of a user query and the associated information needed to generate a valid and personalized travel plan. The core components include:

* **User Queries:** Natural language requests for travel plans, which have also been converted into a structured JSON format. These queries specify constraints such as origin, destination, dates, budget, accommodation type, and cuisine preferences.
* **User Profiles:** Inferred from users' review histories, capturing their likes and dislikes about various POI features.
* **POI and Travel Data:** A comprehensive database of hotels, restaurants, and attractions with associated metadata.

## Citation

Please cite the relevant papers below.

```bibtex
@misc{xie2024travelplanner,
    title={Travelplanner: A benchmark for real-world planning with language agents},
    author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
    year={2024},
    eprint={2402.01622},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{yan2023personalized,
    title={Personalized showcases: Generating multi-modal explanations for recommendations},
    author={Yan, An and He, Zhankui and Li, Jiacheng and Zhang, Tianyang and McAuley, Julian},
    booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={2251--2255},
    year={2023}
}
```
