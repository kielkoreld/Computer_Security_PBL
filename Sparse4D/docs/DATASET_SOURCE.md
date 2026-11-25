---
license: apache-2.0
viewer: false
---
# **Bench2Drive**: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving.

## New Features

This is the new 0.0.4 version of Bench2Drive dataset with strictly uniform training data: The new Think2Drive-collected training set (44 scenarios × 25 routes = 1,100 routes) is strictly uniform over scenarios, unlike the previous Base set whose scenario distribution is uneven.

Also introduced new modality: 3D occupancy (3D occ).

**This repo does not include the depth part of bench2drive v0.0.4 dataset.**

## Description

Bench2Drive is a benchmark designed for evaluating end-to-end autonomous driving algorithms in the closed-loop manner. It features:
- **Comprehensive Scenario Coverage**: Bench2Drive is designed to test AD systems across 44 interactive scenarios, ensuring a thorough evaluation of an AD system's capability to handle real-world driving challenges.
- **Granular Skill Assessment**: By structuring the evaluation across 220 short routes, each focusing on a specific driving scenario, Bench2Drive allows for detailed analysis and comparison of how different AD systems perform on individual tasks.
- **Closed-Loop Evaluation Protocol**: Bench2Drive evaluates AD systems in a closed-loop manner, where the AD system's actions directly influence the environment. This setup offers an accurate assessment of AD systems' driving performance.
- **Diverse Large-Scale Official Training Data**: Bench2Drive consists of a standardized training set of 10000 fully annotated clips under diverse scenarios, weathers, and towns, ensuring that all AD systems are trained under abundant yet similar conditions, which is crucial for fair algorithm-level comparisons.

**Each clip named by: ScenarioName_TownID_RouteID_WeatherID_TimeStamp.tar.gz.**

For depth, please refer to https://huggingface.co/datasets/Bench2DriveData/Bench2Drive-V0.0.4-depth.

For HD-map, please refer to https://huggingface.co/datasets/rethinklab/Bench2Drive-Map-V0.0.4.

For old base1000 set, please refer to https://huggingface.co/datasets/rethinklab/Bench2Drive.

For full set, please refer to https://huggingface.co/datasets/rethinklab/Bench2Drive-Full.

For more information, please visit our GitHub repository: https://github.com/Thinklab-SJTU/Bench2Drive.


## License and Citation

All assets and code are under the Apache 2.0 license unless specified otherwise.

```bibtex
@article{jia2024bench,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Jiazi Bu and Junchi Yan},
  journal={\url{https://github.com/Thinklab-SJTU/Bench2Drive}},
  year={2024}
}
```
## Source of the included examples

The bundled camera-input examples, converted pose metadata, and detector-output examples originate from the Bench2Drive V0.0.4 dataset above, revision `d84ae18cf20a88b9220df1bc15c073d8824b7ac1`.

Camera images have been resized/cropped and stored as model input arrays. Coordinates and annotations have been transformed into the described model/world representations, and predictions are generated model outputs. These are not an unchanged full dataset distribution.

The data card for this specific Hugging Face distribution states Apache-2.0. Its license text is retained in `DATASET_LICENSE.txt`. This statement applies to the referenced dataset distribution; it does not relicense separately distributed Bench2Drive project code or third-party components.
