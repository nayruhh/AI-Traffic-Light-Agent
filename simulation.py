"""
2D Pygame visualization for the TrafficEnvironment.

Renders a top-down 4-way intersection showing:
- Queue lengths as car blocks in each lane
- Traffic light boxes (red/green) for each direction
- Green-tinted road surface for the active direction
- Emergency vehicles that blink yellow/red with a label
- Lane labels (N, S, E, W) with queue counts
- Side panel with stats and a color legend

Controls:
    1-4   -> switch agent
    SPACE -> pause / resume
    R     -> restart episode
    ESC   -> quit
"""
from __future__ import annotations

import numpy as np
import pygame
from traffic_environment import TrafficEnvironment
from fixed_timer_agent import FixedTimerAgent
from queue_threshold_agent import QueueThresholdAgent


# ── Agent wrappers ─────────────────────────────────────────────────────

class RandomAgent:
    """Picks action 0 or 1 at random each step."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def choose_action(self, state):
        return int(self.rng.integers(0, 2))

AGENT_CHOICES = [
    ("Queue Threshold", lambda: QueueThresholdAgent()),
    ("Fixed Timer (10)", lambda: FixedTimerAgent(switch_interval=10)),
    ("Random",           lambda: RandomAgent(seed=0)),
]

# ── Window & layout constants ──────────────────────────────────────────
INTERSECTION_SIZE = 600          # left portion: intersection view
PANEL_W = 220                    # right portion: stats panel
WIDTH = INTERSECTION_SIZE + PANEL_W
HEIGHT = 600
FPS = 2  # slow: easy to follow each step

# Colors
BG_COLOR = (30, 30, 35)
ROAD_COLOR = (70, 70, 75)
ROAD_GREEN_TINT = (45, 75, 50)  # subtle green overlay on active road
INTERSECTION_COLOR = (90, 90, 95)
LINE_COLOR = (120, 120, 120)
CAR_COLOR = (80, 150, 240)
CAR_BORDER = (50, 100, 180)
EMERGENCY_A = (255, 210, 40)    # blink color A (bright yellow)
EMERGENCY_B = (240, 60, 60)     # blink color B (red)
EMERGENCY_BORDER = (180, 50, 50)
GREEN = (30, 200, 60)
RED = (210, 40, 40)
LIGHT_OFF = (60, 60, 60)        # inactive bulb color
LIGHT_HOUSING = (25, 25, 25)    # traffic light box
TEXT_COLOR = (220, 220, 220)
DIM_TEXT = (140, 140, 140)
PANEL_BG = (38, 38, 44)
PANEL_BORDER = (60, 60, 70)
LABEL_BG = (50, 50, 58)

# Road geometry
ROAD_W = 80
CX, CY = INTERSECTION_SIZE // 2, HEIGHT // 2  # intersection center
CAR_W, CAR_H = 14, 18           # car block dimensions
CAR_GAP = 4

EMERGENCY_LANE_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}
LANE_LABELS = ["N", "S", "E", "W"]


# ── Drawing helpers ────────────────────────────────────────────────────

def draw_roads(surface: pygame.Surface, phase: int) -> None:
    """Draw roads with a green tint on the active direction."""
    half = ROAD_W // 2

    # Determine which road arms get the green tint
    ns_color = ROAD_GREEN_TINT if phase == 0 else ROAD_COLOR
    ew_color = ROAD_GREEN_TINT if phase == 1 else ROAD_COLOR

    # Vertical road (N-S)
    pygame.draw.rect(surface, ns_color, (CX - half, 0, ROAD_W, HEIGHT))
    # Horizontal road (E-W)
    pygame.draw.rect(surface, ew_color, (0, CY - half, INTERSECTION_SIZE, ROAD_W))
    # Center intersection
    pygame.draw.rect(surface, INTERSECTION_COLOR,
                     (CX - half, CY - half, ROAD_W, ROAD_W))

    # Dashed lane dividers
    for y in range(0, HEIGHT, 24):
        pygame.draw.line(surface, LINE_COLOR, (CX, y), (CX, y + 10), 1)
    for x in range(0, INTERSECTION_SIZE, 24):
        pygame.draw.line(surface, LINE_COLOR, (x, CY), (x + 10, CY), 1)


def draw_traffic_light_box(
    surface: pygame.Surface,
    x: int, y: int,
    is_green: bool,
    vertical: bool = True,
) -> None:
    """
    Draw a small traffic light housing with a red and green bulb.
    The active bulb is bright; the other is dim.
    """
    bulb_r = 7
    if vertical:
        box = pygame.Rect(x - 12, y - 18, 24, 36)
        red_pos = (x, y - 8)
        green_pos = (x, y + 8)
    else:
        box = pygame.Rect(x - 18, y - 12, 36, 24)
        red_pos = (x - 8, y)
        green_pos = (x + 8, y)

    pygame.draw.rect(surface, LIGHT_HOUSING, box, border_radius=4)
    pygame.draw.rect(surface, (80, 80, 80), box, width=1, border_radius=4)

    red_color = RED if not is_green else LIGHT_OFF
    green_color = GREEN if is_green else LIGHT_OFF
    pygame.draw.circle(surface, red_color, red_pos, bulb_r)
    pygame.draw.circle(surface, green_color, green_pos, bulb_r)


def draw_traffic_lights(surface: pygame.Surface, phase: int) -> None:
    """Place traffic light boxes at each side of the intersection."""
    half = ROAD_W // 2
    gap = 22  # distance from intersection edge

    ns_green = phase == 0
    ew_green = phase == 1

    # North light (above intersection)
    draw_traffic_light_box(surface, CX - half - gap, CY - half - gap,
                           ns_green, vertical=True)
    # South light (below intersection)
    draw_traffic_light_box(surface, CX + half + gap, CY + half + gap,
                           ns_green, vertical=True)
    # East light (right of intersection)
    draw_traffic_light_box(surface, CX + half + gap, CY - half - gap,
                           ew_green, vertical=False)
    # West light (left of intersection)
    draw_traffic_light_box(surface, CX - half - gap, CY + half + gap,
                           ew_green, vertical=False)


def draw_queue(
    surface: pygame.Surface,
    lane_index: int,
    count: int,
    is_emergency: bool,
    frame: int,
) -> None:
    """Draw car blocks queued in the given lane. Emergency car blinks."""
    half = ROAD_W // 2
    step = CAR_H + CAR_GAP

    for i in range(count):
        # Emergency vehicle is the last car in the queue
        is_em_car = is_emergency and i == count - 1

        if is_em_car:
            # Blink between two colors every other frame
            color = EMERGENCY_A if frame % 2 == 0 else EMERGENCY_B
            border = EMERGENCY_BORDER
        else:
            color = CAR_COLOR
            border = CAR_BORDER

        # Position depends on lane direction
        if lane_index == 0:  # North — queue upward
            x = CX - CAR_W // 2 - 12
            y = CY - half - step * (i + 1)
            w, h = CAR_W, CAR_H
        elif lane_index == 1:  # South — queue downward
            x = CX - CAR_W // 2 + 12
            y = CY + half + step * i + CAR_GAP
            w, h = CAR_W, CAR_H
        elif lane_index == 2:  # East — queue rightward
            x = CX + half + step * i + CAR_GAP
            y = CY - CAR_H // 2 - 12
            w, h = CAR_H, CAR_W  # rotated
        else:  # West — queue leftward
            x = CX - half - step * (i + 1)
            y = CY - CAR_H // 2 + 12
            w, h = CAR_H, CAR_W

        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, border, rect, width=1, border_radius=3)

        # Draw "!" label on emergency vehicle
        if is_em_car:
            em_font = pygame.font.SysFont("monospace", 13, bold=True)
            label = em_font.render("!", True, (0, 0, 0))
            lx = x + (w - label.get_width()) // 2
            ly = y + (h - label.get_height()) // 2
            surface.blit(label, (lx, ly))


def draw_lane_labels(
    surface: pygame.Surface,
    font: pygame.font.Font,
    queues: tuple,
) -> None:
    """Draw lane direction labels and queue counts at the road ends."""
    half = ROAD_W // 2
    q_N, q_S, q_E, q_W = queues

    labels = [
        ("N", f"{q_N}", CX, 18),           # top
        ("S", f"{q_S}", CX, HEIGHT - 18),   # bottom
        ("E", f"{q_E}", INTERSECTION_SIZE - 22, CY),  # right
        ("W", f"{q_W}", 22, CY),             # left
    ]

    bold_font = pygame.font.SysFont("monospace", 18, bold=True)
    count_font = pygame.font.SysFont("monospace", 14)

    for name, count_str, lx, ly in labels:
        # Direction letter
        lbl = bold_font.render(name, True, TEXT_COLOR)
        # Queue count
        cnt = count_font.render(count_str, True, DIM_TEXT)

        if name in ("N", "S"):
            # Stack vertically: label then count
            total_h = lbl.get_height() + 2 + cnt.get_height()
            start_y = ly - total_h // 2
            surface.blit(lbl, (lx - lbl.get_width() // 2, start_y))
            surface.blit(cnt, (lx - cnt.get_width() // 2,
                               start_y + lbl.get_height() + 2))
        else:
            # Stack horizontally: label then count
            total_w = lbl.get_width() + 4 + cnt.get_width()
            start_x = lx - total_w // 2
            cy_off = ly - lbl.get_height() // 2
            surface.blit(lbl, (start_x, cy_off))
            surface.blit(cnt, (start_x + lbl.get_width() + 4,
                               ly - cnt.get_height() // 2))


def draw_panel(
    surface: pygame.Surface,
    state: tuple,
    reward: float,
    step: int,
    cumulative_reward: float,
    paused: bool,
    agent_index: int,
) -> None:
    """Draw the right-side stats panel with legend."""
    q_N, q_S, q_E, q_W, phase, emergency_lane = state
    panel_x = INTERSECTION_SIZE

    # Panel background
    pygame.draw.rect(surface, PANEL_BG,
                     (panel_x, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surface, PANEL_BORDER,
                     (panel_x, 0), (panel_x, HEIGHT), 2)

    title_font = pygame.font.SysFont("monospace", 16, bold=True)
    font = pygame.font.SysFont("monospace", 13)
    small_font = pygame.font.SysFont("monospace", 11)

    x = panel_x + 16
    y = 16

    # ── Title ──
    title = title_font.render("TRAFFIC SIM", True, TEXT_COLOR)
    surface.blit(title, (x, y))
    y += 28

    # ── Agent selector ──
    agent_title = font.render("Agent:", True, DIM_TEXT)
    surface.blit(agent_title, (x, y))
    y += 18
    for i, (name, _) in enumerate(AGENT_CHOICES):
        prefix = ">" if i == agent_index else " "
        color = GREEN if i == agent_index else DIM_TEXT
        lbl = small_font.render(f"{prefix} {i+1}. {name}", True, color)
        surface.blit(lbl, (x + 4, y))
        y += 16
    y += 8

    # ── Status section ──
    phase_str = "N-S GREEN" if phase == 0 else "E-W GREEN"
    em_names = {0: "None", 1: "North", 2: "South", 3: "East", 4: "West"}

    rows = [
        ("Step", str(step)),
        ("Phase", phase_str),
        ("Reward", f"{reward:.0f}"),
        ("Total", f"{cumulative_reward:.0f}"),
        ("Emergency", em_names[emergency_lane]),
    ]

    for label, value in rows:
        lbl_surf = font.render(f"{label}:", True, DIM_TEXT)
        val_surf = font.render(value, True, TEXT_COLOR)
        surface.blit(lbl_surf, (x, y))
        surface.blit(val_surf, (x + 100, y))
        y += 20

    # ── Queue bars ──
    y += 10
    bar_title = title_font.render("QUEUES", True, TEXT_COLOR)
    surface.blit(bar_title, (x, y))
    y += 24

    max_bar_w = PANEL_W - 80
    max_q = 10
    for name, q_val in [("N", q_N), ("S", q_S), ("E", q_E), ("W", q_W)]:
        lbl = font.render(f"{name}:", True, DIM_TEXT)
        surface.blit(lbl, (x, y))
        bar_x = x + 30
        bar_w = int((q_val / max_q) * max_bar_w) if max_q > 0 else 0
        pygame.draw.rect(surface, (50, 50, 55),
                         (bar_x, y + 2, max_bar_w, 14), border_radius=2)
        if bar_w > 0:
            bar_color = (240, 80, 80) if q_val >= 8 else CAR_COLOR
            pygame.draw.rect(surface, bar_color,
                             (bar_x, y + 2, bar_w, 14), border_radius=2)
        cnt = font.render(str(q_val), True, TEXT_COLOR)
        surface.blit(cnt, (bar_x + max_bar_w + 6, y))
        y += 22

    # ── Legend ──
    y += 12
    legend_title = title_font.render("LEGEND", True, TEXT_COLOR)
    surface.blit(legend_title, (x, y))
    y += 24

    legend_items = [
        (CAR_COLOR, "Normal car"),
        (EMERGENCY_A, "Emergency (!)"),
        (GREEN, "Green light"),
        (RED, "Red light"),
        (ROAD_GREEN_TINT, "Active direction"),
    ]

    for color, desc in legend_items:
        pygame.draw.rect(surface, color,
                         (x, y + 1, 14, 14), border_radius=2)
        pygame.draw.rect(surface, (100, 100, 100),
                         (x, y + 1, 14, 14), width=1, border_radius=2)
        lbl = small_font.render(desc, True, DIM_TEXT)
        surface.blit(lbl, (x + 22, y + 1))
        y += 20

    # ── Controls ──
    y = HEIGHT - 70
    pygame.draw.line(surface, PANEL_BORDER,
                     (panel_x + 10, y), (panel_x + PANEL_W - 10, y), 1)
    y += 10
    controls = [
        "1-3    switch agent",
        "SPACE  pause/resume",
        "R      restart",
        "ESC    quit",
    ]
    for line in controls:
        lbl = small_font.render(line, True, DIM_TEXT)
        surface.blit(lbl, (x, y))
        y += 14

    # ── Paused overlay ──
    if paused:
        pause_font = pygame.font.SysFont("monospace", 22, bold=True)
        pause_lbl = pause_font.render("PAUSED", True, EMERGENCY_A)
        px = panel_x + (PANEL_W - pause_lbl.get_width()) // 2
        surface.blit(pause_lbl, (px, HEIGHT // 2))


# ── Main loop ──────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Intersection Simulation")
    clock = pygame.time.Clock()

    env = TrafficEnvironment(
        arrival_rate_ns=0.4,   # balanced: arrivals ≤ departure capacity
        arrival_rate_ew=0.4,
        depart_per_lane=1,
        emergency_prob=0.05,
        seed=42,
    )
    state = env.reset()
    reward = 0.0
    cumulative_reward = 0.0
    step_count = 0
    frame = 0
    paused = False
    running = True

    # Start with the Queue Threshold agent (index 0)
    agent_index = 0
    agent = AGENT_CHOICES[agent_index][1]()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Restart the episode
                    state = env.reset()
                    step_count = 0
                    cumulative_reward = 0.0
                    reward = 0.0
                    agent = AGENT_CHOICES[agent_index][1]()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    # Switch agent
                    new_index = event.key - pygame.K_1  # 0, 1, or 2
                    if new_index < len(AGENT_CHOICES):
                        agent_index = new_index
                        agent = AGENT_CHOICES[agent_index][1]()

        if not paused and step_count < env.max_steps:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            cumulative_reward += reward
            step_count += 1
            if done:
                state = env.reset()
                step_count = 0
                cumulative_reward = 0.0

        # ── Draw ──
        screen.fill(BG_COLOR)

        q_N, q_S, q_E, q_W, phase, emergency_lane = state

        draw_roads(screen, phase)
        draw_traffic_lights(screen, phase)

        for lane_idx, count in enumerate([q_N, q_S, q_E, q_W]):
            is_em = (
                emergency_lane != 0
                and EMERGENCY_LANE_TO_INDEX.get(emergency_lane) == lane_idx
            )
            draw_queue(screen, lane_idx, count, is_em, frame)

        draw_lane_labels(screen, None, (q_N, q_S, q_E, q_W))
        draw_panel(screen, state, reward, step_count,
                   cumulative_reward, paused, agent_index)

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()


if __name__ == "__main__":
    main()
