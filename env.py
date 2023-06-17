import math

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

try:
    import pygame
    from pygame import Rect
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, please install pygame."
    ) from e


class SkillCheck(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 109}

    def __init__(self, render_mode=None):
        self.cycle_length = 120
        self.good_success_zone_angle = 45.0
        self.great_success_zone_angle = 10.0

        self.speed = 360. / self.cycle_length

        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

        # Rendering configuration
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.screen_width = 300
        self.screen_height = 300
        self.state_width = 84
        self.state_height = 84
        self.screen = None
        self.clock = None

    def reset(self, seed=None):
        super().reset(seed=seed)

        p1 = np.random.randint(0, 210)
        p2 = p1 + self.great_success_zone_angle
        p3 = p2 + self.good_success_zone_angle
        self._state = {'t': 0, 'p1': p1, 'p2': p2, 'p3': p3, 'status': 0}

        observation = self._render('state_pixels')
        info = dict()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        reward = 0
        terminated = False
        p = self.speed * self._state['t'] - 90
        if action == 1:
            if p < self._state['p1']:
                reward = -5.0
                self._state['status'] = 1

            elif (p >= self._state['p1']) and (p < self._state['p2']):
                reward = 10.0
                self._state['status'] = 2

            elif (p >= self._state['p2']) and (p < self._state['p3']):
                reward = 1.0
                self._state['status'] = 3
            terminated = True

        if p + self.speed >= self._state['p3']:
            reward = -5.0
            self._state['status'] = 1
            terminated = True

        self._state['t'] += 1
        observation = self._render('state_pixels')
        info = dict()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _render(self, mode):
        screen_width, screen_height = self.screen_width, self.screen_height
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))

        xc, yc = screen_width / 2.0, screen_height / 2.0
        radius = screen_width / 2.2
        offset = 6
        white = (255, 255, 255)

        t1_rad = self._state['p1'] * math.pi / 180.0
        t2_rad = self._state['p2'] * math.pi / 180.0
        t3_rad = self._state['p3'] * math.pi / 180.0

        pygame.draw.arc(
            self.surf,
            color=white,
            rect=pygame.Rect(xc - radius, yc - radius, 2 * radius, 2 * radius),
            start_angle=t3_rad,
            stop_angle=t1_rad,
            width=2,
        )
        pygame.draw.arc(
            self.surf,
            color=white,
            rect=Rect(xc - radius - offset, yc - radius - offset, 2 * radius + 2 * offset, 2 * radius + 2 * offset),
            start_angle=t1_rad,
            stop_angle=t3_rad,
            width=2,
        )
        pygame.draw.arc(
            self.surf,
            color=white,
            rect=Rect(xc - radius + offset, yc - radius + offset, 2 * radius - 2 * offset, 2 * radius - 2 * offset),
            start_angle=t1_rad,
            stop_angle=t3_rad,
            width=2
        )
        pygame.draw.line(
            self.surf,
            color=white,
            start_pos=(yc + (radius - offset) * math.cos(t1_rad), xc - (radius - offset) * math.sin(t1_rad)),
            end_pos=(yc + (radius + offset) * math.cos(t1_rad), xc - (radius + offset) * math.sin(t1_rad)),
            width=2
        )
        pygame.draw.line(
            self.surf,
            color=white,
            start_pos=(yc + (radius - offset) * math.cos(t2_rad), xc - (radius - offset) * math.sin(t2_rad)),
            end_pos=(yc + (radius + offset) * math.cos(t2_rad), xc - (radius + offset) * math.sin(t2_rad)),
            width=2
        )
        pygame.draw.line(
            self.surf,
            color=white,
            start_pos=(yc + (radius - offset) * math.cos(t3_rad), xc - (radius - offset) * math.sin(t3_rad)),
            end_pos=(yc + (radius + offset) * math.cos(t3_rad), xc - (radius + offset) * math.sin(t3_rad)),
            width=2
        )

        p_rad = (self.speed * self._state['t'] - 90) * math.pi / 180.
        pygame.draw.line(
            self.surf,
            color=(255, 0, 0),
            start_pos=(xc, yc),
            end_pos=(yc + (radius + 3 * offset) * math.cos(p_rad), xc - (radius + 3 * offset) * math.sin(p_rad)),
            width=4
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        if mode == 'human':
            if self._state['status'] == 1:
                self.surf.fill((255, 0, 0))
            elif self._state['status'] == 2:
                self.surf.fill((0, 0, 255))
            elif self._state['status'] == 3:
                self.surf.fill((0, 255, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == 'rgb_array':
            if self._state['status'] == 1:
                self.surf.fill((255, 0, 0))
            elif self._state['status'] == 2:
                self.surf.fill((0, 0, 255))
            elif self._state['status'] == 3:
                self.surf.fill((0, 255, 0))
            return self._create_image_array(self.surf, (self.screen_width, self.screen_height))
        elif mode == 'state_pixels':
            return self._create_image_array(self.surf, (self.state_width, self.state_height))

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specfifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), (1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        return img

    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)

        # Convert a frame to 84 X 84 gray scale one
        s = self._preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info

    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = self._preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info


if __name__ == '__main__':
    env = SkillCheck(render_mode='human')
    quit = False
    while not quit:
        env.reset()
        a = 0
        total_reward = 0.0
        steps = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        a = 1
                if event.type == pygame.KEYUP:
                    a = 0
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r

            if terminated or truncated or quit:
                break
        print(total_reward)
    env.close()
