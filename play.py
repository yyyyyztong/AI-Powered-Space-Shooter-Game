import random
import pygame
from pygame import *
import time
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, q_table_file="q_table.pkl"):
        self.actions = actions  # Action space
        self.lr = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table, stored as a dictionary
        self.q_table_file = q_table_file  # File to save the Q-table
        self.load_q_table()

    def save_q_table(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved to file.")

    def load_q_table(self):
        try:
            with open(self.q_table_file, "rb") as f:
                self.q_table = pickle.load(f)
            print("Q-table loaded from file.")
        except FileNotFoundError:
            print("No Q-table file found, starting with an empty Q-table.")

    def get_state(self, player, enemy_bullets, enemies):
        def discretize(value, step=10):
            return value // step  

        if enemy_bullets:
            closest_bullet = min(enemy_bullets, key=lambda b: abs(b.rect.top - player.rect.top))
            distance_bullet_x = discretize(abs(player.rect.left - closest_bullet.rect.left))
            distance_bullet_y = discretize(abs(player.rect.top - closest_bullet.rect.top))
        else:
            distance_bullet_x, distance_bullet_y = 0, 0

        if enemies:
            closest_enemy = min(enemies, key=lambda e: abs(e.rect.top - player.rect.top))
            distance_enemy_x = discretize(abs(player.rect.left - closest_enemy.rect.left))
            distance_enemy_y = discretize(abs(player.rect.top - closest_enemy.rect.top))
        else:
            distance_enemy_x, distance_enemy_y = 0, 0

        return (distance_bullet_x, distance_bullet_y, distance_enemy_x, distance_enemy_y)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return np.random.choice(self.actions)  # choose action randomly
        return max(self.q_table[state], key=self.q_table[state].get)  # choose the best action

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * max(self.q_table[next_state].values())
        self.q_table[state][action] += self.lr * (q_target - q_predict)

    def update_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate) 

class Player(pygame.sprite.Sprite):
    bullets = pygame.sprite.Group()

    def __init__(self, screen, agent):
        pygame.sprite.Sprite.__init__(self)

        # load player image
        self.player = pygame.image.load("./images/me1.png")
        self.rect = self.player.get_rect()
        self.rect.topleft = [240 - 51, 550]

        self.speed = 5
        self.screen = screen
        self.bullets = pygame.sprite.Group()
        self.last_shot_time = 0
        self.shoot_cooldown = 100
        self.agent = agent  # Q-learning
        self.current_action = None  # current action

    def auto_control(self, enemy_bullets, enemies):
        state = self.agent.get_state(self, enemy_bullets, enemies)
        self.current_action = self.agent.choose_action(state)

        if self.current_action == "UP":
            self.rect.top -= self.speed
        elif self.current_action == "DOWN":
            self.rect.bottom += self.speed
        elif self.current_action == "LEFT":
            self.rect.left -= self.speed
        elif self.current_action == "RIGHT":
            self.rect.right += self.speed
        elif self.current_action == "AVOID":  # avoid the bullets
            if enemy_bullets:
                closest_bullet = min(enemy_bullets, key=lambda b: abs(b.rect.top - self.rect.top))
                if self.rect.left < closest_bullet.rect.left:  
                    self.rect.left -= self.speed  
                else:  
                    self.rect.left += self.speed 
                if self.rect.top < closest_bullet.rect.top:  
                    self.rect.top -= self.speed  
                else:  
                    self.rect.top += self.speed 
        elif self.current_action == "CHASE":  # chase enemy
            if enemies:
                # Find the enemy aircraft closest to the y-coordinate of the fighter.
                closest_enemy = min(enemies, key=lambda e: abs(e.rect.bottom - self.rect.top))
                
                if abs(self.rect.left - closest_enemy.rect.left) <= self.speed:
                    self.rect.left = closest_enemy.rect.left
                elif self.rect.left < closest_enemy.rect.left:
                    self.rect.left += self.speed
                elif self.rect.left > closest_enemy.rect.left:
                    self.rect.left -= self.speed

        # check borden
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > 700:
            self.rect.bottom = 700
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > 480:
            self.rect.right = 480


    def auto_fire(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > self.shoot_cooldown:  
            bullet = Bullet(self.screen, self.rect.left, self.rect.top)
            self.bullets.add(bullet)
            Player.bullets.add(bullet)
            self.last_shot_time = current_time

    def update(self, enemy_bullets, enemies, reward):
        if not isinstance(reward, (int, float)):
            raise ValueError(f"Reward must be a number, got {type(reward)} instead.")

        self.auto_control(enemy_bullets, enemies)  
        self.auto_fire()
        self.display()

        # update q table
        next_state = self.agent.get_state(self, enemy_bullets, enemies)
        self.agent.learn(self.agent.get_state(self, enemy_bullets, enemies), self.current_action, reward, next_state)

    def display(self):
        self.screen.blit(self.player, self.rect)
        self.bullets.update()
        self.bullets.draw(self.screen)

    @classmethod
    def clear_bullets(cls):
        cls.bullets.empty()

class Enemy(pygame.sprite.Sprite):
    enemy_bullets = pygame.sprite.Group()
    def __init__(self, screen):

        pygame.sprite.Sprite.__init__(self)
        # load player image
        self.player = pygame.image.load("./images/enemy1.png") # 57 * 43

        self.rect = self.player.get_rect()

        x = random.randrange(1, Manager.bg_size[0], 50)
        self.rect.topleft = [x, 0]

        self.speed = 2

        self.screen = screen

        self.bullets = pygame.sprite.Group()

        self.direction = 'right'

        self.last_shot_time = 0  
        self.shoot_cooldown = 500


    def display(self):
         # draw player at the center of the screen
        self.screen.blit(self.player, self.rect)    
        
        # display bullets
        self.bullets.update()
        self.bullets.draw(self.screen)

    def auto_move(self):
        if self.direction == 'right':
            self.rect.left += 3
        elif self.direction == 'left':
            self.rect.left -= 3

        if self.rect.left < 0:  
            self.rect.left = 0
            self.direction = 'right'
        elif self.rect.left > 480 - self.rect.width: 
            self.rect.left = 480 - self.rect.width
            self.direction = 'left'


        self.rect.bottom += self.speed

    def update(self):
        self.auto_move()
        self.fire_bullet()
        self.display()
    
    def fire_bullet(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > self.shoot_cooldown:  
            bullet = EnemyBullet(self.screen, self.rect.left, self.rect.top)
            self.bullets.add(bullet)
            Enemy.enemy_bullets.add(bullet)
            self.last_shot_time = current_time  

    @classmethod
    def clear_bullets(cls):
        cls.enemy_bullets.empty()

class Bullet(pygame.sprite.Sprite):
    def __init__(self, screen, x, y):

        pygame.sprite.Sprite.__init__(self)

        # load bullet image
        self.image = pygame.image.load("./images/bullet1.png")

        # location
        self.rect = self.image.get_rect()
        self.rect.topleft = [x + 51 - 2, y - 11]

        self.screen = screen

        self.speed = 5

    def update(self):
        # 
        self.rect.top -= self.speed
        if self.rect.top < -22:
            self.kill()

class EnemyBullet(pygame.sprite.Sprite):
    def __init__(self, screen, x, y):

        pygame.sprite.Sprite.__init__(self)

        # load bullet image
        self.image = pygame.image.load("./images/bullet2.png") # 5 * 11

        # location
        self.rect = self.image.get_rect()
        self.rect.topleft = [x + 56/2 - 6/2, y + 43]

        self.screen = screen

        self.speed = 2.5

    def update(self):
        self.rect.top += self.speed
        if self.rect.top > 852:
            self.kill()

class BGM(object):
    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.music.load("./sound/game_music.ogg")
        pygame.mixer.music.set_volume(0.5) # sound

        self.__bomb = pygame.mixer.Sound("./sound/get_bomb.wav")

    def play(self):
        pygame.mixer.music.play(-1)

    def play_bomb(self):
        pygame.mixer.Sound.play(self.__bomb)

class Bomb(object):
    def __init__(self, screen, type):
        self.screen = screen

        if type == 'emeny':
            self.mImages = [pygame.image.load("./images/enemy1_down" + str(v) + '.png') for v in range(1, 5)]
        else:
            self.mImages = [pygame.image.load("./images/me_destroy_" + str(v) + '.png') for v in range(1, 5)]

        self.mIndex = 0
        self.mPos = [0,0]
        self.mVisible = False

    def action(self, rect):
        self.mPos[0] = rect.left
        self.mPos[1] = rect.top
        # bomb
        self.mVisible = True

    def draw(self):
        if not self.mVisible:
            return
        self.screen.blit(self.mImages[self.mIndex], (self.mPos[0], self.mPos[1]))
        self.mIndex += 1
        if self.mIndex >= len(self.mImages):
            self.mIndex = 0
            self.mVisible = False

class Map(object):
    def __init__(self, screen):
        self.mImage1 = pygame.image.load("./images/background.png")
        self.mImage2 = pygame.image.load("./images/background.png")

        # window
        self.screen = screen
        self.y1 = 0
        self.y2 = -self.mImage1.get_height()

    def move(self):
        self.y1 += 2
        self.y2 += 2

        if self.y1 >= self.mImage1.get_height():
            self.y1 = -self.mImage1.get_height()
        if self.y2 >= self.mImage2.get_height():
            self.y2 = -self.mImage2.get_height()
    
    def draw(self):
        self.screen.blit(self.mImage1, (0, self.y1))
        self.screen.blit(self.mImage2, (0, self.y2))

class Manager:
    bg_size = (480, 700)
    create_enemy_id = 10
    game_over_id = 11
    is_game_over = False
    score = 0  # score

    def __init__(self):
        pygame.init()
        # create a window
        self.screen = pygame.display.set_mode(Manager.bg_size, 0, 32)
        # load background image
        self.map = Map(self.screen)
        # init a group for players
        self.players = pygame.sprite.Group()
        # init a group for enemies
        self.enemies = pygame.sprite.Group()
        # bomb
        self.player_bomb = Bomb(self.screen, 'me')
        self.enemy_bomb = Bomb(self.screen, 'emeny')
        # load bgm
        self.sound = BGM()
        self.agent = QLearningAgent(actions=["UP", "DOWN", "LEFT", "RIGHT", "AVOID"])

    def exit(self):
        print("quit")
        pygame.quit()
        exit()

    def new_player(self):
        player = Player(self.screen, self.agent)
        self.players.add(player)

    def new_enemy(self):
        enemy = Enemy(self.screen)
        self.enemies.add(enemy)

    def drawText(self, text,x ,y, textHeight=30, fontColor=(255,0,0), backgroundColor=None):
        font = pygame.font.Font(None, textHeight)
        textImage = font.render(text, True, fontColor, backgroundColor)
        text_rect = textImage.get_rect()
        text_rect.topleft = (x, y)
        self.screen.blit(textImage, text_rect)

    def reset_game(self):
        Manager.is_game_over = False  
        Manager.score = 0 
        self.players.empty() 
        self.enemies.empty() 
        Player.clear_bullets()  
        Enemy.clear_bullets()  
        self.new_player()  

    def train(self, episodes=1000, max_steps=500):
        """
        Train the agent using Q-learning.
        :param episodes: Number of training episodes.
        :param max_steps: Maximum steps per episode.
        """
        # Initialize the Q table
        self.agent.load_q_table()

        for episode in range(episodes):
            # Reset game environment
            self.reset_game()
            # self.new_player()
            pygame.time.set_timer(Manager.create_enemy_id, 1000)

            player = self.players.sprites()[0]
            state = self.agent.get_state(player, self.enemies, Enemy.enemy_bullets)
            total_reward = 0

            for step in range(max_steps):
                self.map.move()
                self.map.draw()

                self.drawText(f'Episode: {episode + 1}/{episodes}', 0, 0)
                self.drawText(f'Score: {Manager.score}', 0, 30)

                reward = 0 
                safe_distance = 40  

                # generate enemy
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # save the q table
                        self.agent.save_q_table()
                        self.exit()
                    elif event.type == Manager.create_enemy_id:
                        self.new_enemy()

                action = self.agent.choose_action(state)

                
                if action == 0: 
                    player.move_up()
                elif action == 1:  
                    player.move_down()
                elif action == 2:  
                    player.move_left()
                elif action == 3: 
                    player.move_right()
                elif action == 4:  
                    player.shoot()

               
                if Enemy.enemy_bullets:
                    closest_bullet = min(Enemy.enemy_bullets, key=lambda b: abs(b.rect.top - player.rect.top))
                    distance_x = abs(player.rect.left - closest_bullet.rect.left)
                    distance_y = abs(player.rect.top - closest_bullet.rect.top)

                    if distance_x < safe_distance and distance_y < safe_distance:  
                        reward -= 10 
                    else:
                        reward += 2  

                # Rewards for hitting enemy aircraft
                is_enemy = pygame.sprite.groupcollide(Player.bullets, self.enemies, True, False)
                if is_enemy:
                    reward += 50  
                    for bullet, enemies in is_enemy.items():
                        for enemy in enemies:
                            self.enemy_bomb.action(enemy.rect)
                            self.sound.play_bomb()
                            Manager.score += 10
                            self.enemies.remove(enemy)
                            reward += 5  # Extra incentives to encourage sustained attacks

                # Check the distance between the player and the enemy aircraft
                if self.enemies:
                    closest_enemy = min(self.enemies, key=lambda e: abs(e.rect.top - player.rect.top))
                    distance_to_enemy = abs(player.rect.top - closest_enemy.rect.top)
                    if distance_to_enemy < safe_distance:  
                        reward += 1  
                    else:
                        reward -= 0.5  

                # Penalty for being hit by enemy aircraft bullets
                if player.rect.top > 5 and player.rect.bottom < 695:
                    isover = pygame.sprite.spritecollide(player, Enemy.enemy_bullets, True)
                    if isover:
                        reward -= 100
                        Manager.is_game_over = True
                        pygame.time.set_timer(Manager.game_over_id, 1000)
                        self.player_bomb.action(player.rect)
                        self.players.remove(player)
                        self.sound.play_bomb()
                        break  

                # Calculate the next state
                next_state = self.agent.get_state(player, self.enemies, Enemy.enemy_bullets)

                # update the next state
                self.agent.learn(state, action, reward, next_state)

                # Update the current state 
                state = next_state

                # Cumulative rewards
                total_reward += reward

                # Updating players and enemy aircraft
                self.players.update(Enemy.enemy_bullets, self.enemies, reward)
                self.enemies.update()

                # Dynamic adjustment of the exploration rate
                self.agent.update_epsilon()

                # pygame.display.update()
                # time.sleep(0.01)

            print(f'Episode {episode + 1}/{episodes} ended with total reward: {total_reward}, score: {Manager.score}')

            self.agent.save_q_table()

        print("Training completed and Q-table saved!")




    def main(self):
        # play music
        self.sound.play()
        self.new_player()
        pygame.time.set_timer(Manager.create_enemy_id, 1000)

        # load q table
        self.agent.load_q_table()

        while True:
            # auto move map
            self.map.move()
            self.map.draw()
            # draw score
            self.drawText(f'Score: {Manager.score}', 0, 0)

            reward = 0  # init reward
            safe_distance = 40  # init the safe distance

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # save q table
                    self.agent.save_q_table()
                    self.exit()
                elif event.type == Manager.create_enemy_id:
                    self.new_enemy()

            player = self.players.sprites()[0]

            # bomb attack
            self.player_bomb.draw()
            self.enemy_bomb.draw()

            # collision detection
            iscollide = pygame.sprite.groupcollide(self.players, self.enemies, True, True)
            if iscollide:
                items = list(iscollide.items())[0]  
                print(items)
                x = items[0]  
                y = items[1][0]  

                # bomb attack
                self.player_bomb.action(x.rect)
                self.enemy_bomb.action(y.rect)
                self.sound.play_bomb()

                # game over
                Manager.is_game_over = True
                pygame.time.set_timer(Manager.game_over_id, 1000)

            if Enemy.enemy_bullets:
                closest_bullet = min(Enemy.enemy_bullets, key=lambda b: abs(b.rect.top - player.rect.top))
                distance_x = abs(player.rect.left - closest_bullet.rect.left)
                distance_y = abs(player.rect.top - closest_bullet.rect.top)

                if distance_x < safe_distance and distance_y < safe_distance:
                    reward -= 10
                else:
                    reward += 2 


            is_enemy = pygame.sprite.groupcollide(Player.bullets, self.enemies, True, False)
            if is_enemy:
                reward += 50  
                for bullet, enemies in is_enemy.items():
                    for enemy in enemies:
                        self.enemy_bomb.action(enemy.rect)
                        self.sound.play_bomb()
                        Manager.score += 10
                        self.enemies.remove(enemy)
                        reward += 5  

        
            if self.enemies:
                closest_enemy = min(self.enemies, key=lambda e: abs(e.rect.top - player.rect.top))
                distance_to_enemy = abs(player.rect.top - closest_enemy.rect.top)
                if distance_to_enemy < safe_distance:  
                    reward += 2  
                else:
                    reward -= 1

          
            if player.rect.top > 5 and player.rect.bottom < 695:
                isover = pygame.sprite.spritecollide(player, Enemy.enemy_bullets, True)
                if isover:
                    reward -= 100
                    Manager.is_game_over = True
                    pygame.time.set_timer(Manager.game_over_id, 1000)
                    self.player_bomb.action(player.rect)
                    self.players.remove(player)
                    self.sound.play_bomb()

            # game over
            if Manager.is_game_over:
                while Manager.is_game_over:  
                    self.map.draw()
                    self.drawText(f'Score: {Manager.score}', 150, 300, 50, (255, 255, 255))
                    
                    
                    button_rect = pygame.Rect(150, 400, 200, 50)  
                    pygame.draw.rect(self.screen, (0, 255, 0), button_rect) 
                    self.drawText("Restart", 180, 410, 40, (0, 0, 0)) 
                    # check if player is dead
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.exit()
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = event.pos
                            if button_rect.collidepoint(mouse_pos):  
                                self.reset_game() 

                    pygame.display.update()
                    time.sleep(0.01)
                    
            self.players.update(Enemy.enemy_bullets, self.enemies, reward) 
            self.enemies.update()
            self.agent.update_epsilon()
            pygame.display.update()
            time.sleep(0.01)


if __name__ == "__main__":
    manager = Manager()
    # Choose whether to train or run the game
    mode = input("Enter 'train' to train AI or 'play' to test AI: ").strip().lower()
    if mode == "train":
        manager.train(episodes=100, max_steps=1000)  # train 1000 times
    elif mode == "play":
        manager.main() 
    else:
        print("Invalid mode. Please enter 'train' or 'play'.")