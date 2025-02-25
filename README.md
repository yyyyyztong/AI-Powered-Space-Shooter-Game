# README: AI-Powered Space Shooter Game

## Overview

This project is a **space shooter game** powered by **Q-learning**, a reinforcement learning algorithm. The game allows an AI agent to control a player spaceship, learn to avoid enemy bullets, chase enemies, and maximize its score by shooting down enemy ships. The game can be played manually or trained in an automated mode.

---

## Features

- **AI-Powered Gameplay**: The player spaceship is controlled by a Q-learning agent that learns from the environment and improves its performance over time.
- **Manual/Training Modes**: Choose between training the AI or testing its performance in real-time gameplay.
- **Dynamic Environment**: Includes moving enemies, player and enemy bullets, and a scrolling background.
- **Reward System**: The AI learns through a reward mechanism, encouraging survival, attacking enemies, and avoiding bullets.
- **Q-Table Persistence**: Saves and loads the Q-table to allow the AI to retain its knowledge across sessions.
- **Interactive Restart**: A restart button allows players to reset the game after it ends.

---

## Prerequisites

Before running the game, ensure the following dependencies are installed:

- **Python 3.8+**
- **Pygame**: `pip install pygame`
- **NumPy**: `pip install numpy`

---

## How to Run

1. Clone the repository or download the source code.
2. Place the required assets (images and sounds) in the appropriate directories:
   - Images: `./images/` (e.g., spaceship, enemy, bullets, background)
   - Sounds: `./sound/` (e.g., background music, explosion sounds)
3. Run the script:
   ```bash
   python <script_name>.py
   ```
4. Choose a mode:
   - **Train**: Train the AI agent using reinforcement learning.
   - **Play**: Test the trained AI in real-time gameplay.

---

## Game Controls

The game is AI-driven, but you can observe the following actions taken by the AI:

- **UP**: Move the player spaceship up.
- **DOWN**: Move the player spaceship down.
- **LEFT**: Move the player spaceship left.
- **RIGHT**: Move the player spaceship right.
- **AVOID**: Avoid incoming bullets by moving strategically.
- **CHASE**: Chase and align with enemies to shoot them.

---

## AI Training

### Q-Learning Overview

The AI uses **Q-learning**, a reinforcement learning algorithm, to make decisions. The agent learns by interacting with the environment and updating its **Q-table**, which maps states to actions.

Key parameters:
- **Learning Rate (α)**: Controls how much new information overrides old information.
- **Discount Factor (γ)**: Determines the importance of future rewards.
- **Exploration Rate (ε)**: Balances exploration (random actions) and exploitation (choosing the best-known action).

### Training Process

1. **States**: The AI observes the environment, including:
   - Distance to the nearest enemy.
   - Distance to the nearest bullet.
   - Relative positions of enemies and bullets.
2. **Actions**: The agent chooses one of the predefined actions (e.g., move, fire, avoid).
3. **Rewards**: The AI receives rewards or penalties based on its decisions:
   - Positive rewards for hitting enemies or maintaining a safe distance.
   - Negative rewards for getting hit by bullets or colliding with enemies.
4. **Q-Table Update**: The Q-table is updated after each action using the Q-learning formula.

### Training Command

To train the AI:
```bash
python <script_name>.py
```
Select `train` mode when prompted. You can adjust the number of episodes and steps in the `train()` method.

---

## Gameplay

### Playing with AI

1. Run the script:
   ```bash
   python <script_name>.py
   ```
2. Select `play` mode when prompted.
3. Watch the AI-controlled spaceship avoid bullets, chase enemies, and maximize its score.

---

## File Structure

- **Main Script**: Contains the game logic, including training and gameplay modes.
- **Images**: Visual assets for the player, enemies, bullets, and background.
- **Sounds**: Audio assets for background music and sound effects.
- **Q-Table**: Saved in a `.pkl` file for persistent learning.

---

## Reward System

The reward system is designed to guide the AI's learning process:
- **+50**: Hitting an enemy.
- **+5**: Additional reward for destroying enemies.
- **+2**: Maintaining a safe distance from bullets.
- **-10**: Getting too close to bullets.
- **-100**: Getting hit by a bullet or colliding with an enemy.

---

## Customization

- **Adjust AI Parameters**: Modify learning rate, discount factor, and exploration rate in the `QLearningAgent` class.
- **Change Rewards**: Update the reward system in the `train()` and `main()` methods.
- **Add Features**: Enhance the game with new mechanics, such as power-ups or advanced enemy behaviors.

---

## Troubleshooting

1. **Missing Assets**: Ensure all required images and sounds are in the correct directories.
2. **Dependency Errors**: Install missing libraries using `pip`.
3. **Performance Issues**: Reduce the number of enemies or bullets to improve performance on slower systems.

---

## Future Improvements

- Implement **Deep Q-Learning** for more complex decision-making.
- Add **multiplayer mode** for competitive gameplay.
- Introduce **different enemy types** with unique behaviors.
- Optimize the reward system for faster learning.

---

## License

This project is open-source and available for personal and educational use. Feel free to modify and share it!

---

Enjoy the game and watch the AI evolve into a skilled space pilot! 🚀
