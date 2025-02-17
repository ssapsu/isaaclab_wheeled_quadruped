# Isaac Lab 2.0 - Wheeled Quadruped Custom Task

## Overview

This repository is a modified version of the **Wheeled Quadruped** code originally developed by Robotmania to run in the Isaac Lab 2.0 environment.
[Isaac Lab Wheeled Quadruped Overview](https://www.youtube.com/watch?v=o9Bym5mOl2k&t=1s)

## Installation & Execution

1. **Clone the Repository**

   Run the following command to clone the repository into the `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/` directory:

   ```bash
   cd IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/
   git clone https://github.com/ssapsu/isaaclab_wheeled_quadruped.git
   ```

2. **Run Training**

    To train the model using reinforcement learning, execute the following command:

    ```bash
    python scripts/reinforcement_learning/sb3/train.py --task Custom-Wheeled-Quadruped-v0 --num_envs 1024
    ```

## References

- **YouTube Reference:**
  [Isaac Lab Wheeled Quadruped Demo Video](https://youtu.be/_79nZS3ey2U)

## Notes

- This project is configured to run in Isaac Lab 2.0.
- The `Custom-Wheeled-Quadruped-v0` environment is used to train a wheeled quadruped robot.
- The number of environments (`--num_envs`) can be adjusted based on your hardware capabilities.
