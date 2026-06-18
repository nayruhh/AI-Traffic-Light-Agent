"""
Automated evaluation pipeline for the Adaptive Traffic Light Control project.

Trains the Hybrid Q-Learning agent and compares it against baselines across
multiple traffic scenarios.  Produces:
    - Console summary table
    - Bar charts (reward, avg queue, emergency waits, throughput)
    - Learning curve plot
    - All figures saved to  results/  directory

Usage:
    python3 evaluate.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — works without a display
import matplotlib.pyplot as plt

from traffic_environment import TrafficEnvironment
from q_learning_agent import QLearningAgent
from fixed_timer_agent import FixedTimerAgent
from queue_threshold_agent import QueueThresholdAgent
from hybrid_agent import HybridAgent, train_hybrid, compress_state

# ── Configuration ──────────────────────────────────────────────────────
N_EVAL = 500          # evaluation episodes per scenario
RESULTS_DIR = "results"

SCENARIOS = {
    "Balanced\n(0.4 / 0.4)": dict(ns=0.4, ew=0.4, em=0.05),
    "Asymmetric\n(0.6 / 0.2)": dict(ns=0.6, ew=0.2, em=0.05),
    "Rush Hour\n(0.7 / 0.7)": dict(ns=0.7, ew=0.7, em=0.05),
    "Emergency\nHeavy": dict(ns=0.4, ew=0.4, em=0.15),
    "Strongly Asym.\n(0.8 / 0.15)": dict(ns=0.8, ew=0.15, em=0.05),
}


# ── Baseline agents ────────────────────────────────────────────────────

class AlwaysKeepAgent:
    """Never switches — keeps the initial phase forever."""
    def choose_action(self, state):
        return 0

class RandomAgent:
    """Picks keep or switch with equal probability."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def choose_action(self, state):
        return int(self.rng.integers(0, 2))


# ── Evaluation helpers ─────────────────────────────────────────────────

def evaluate_agent(agent, ns_rate, ew_rate, em_prob, n_eval=N_EVAL):
    """Run n_eval episodes and return per-episode metrics."""
    rewards, avg_queues, emg_waits, throughputs = [], [], [], []
    for i in range(n_eval):
        env = TrafficEnvironment(
            arrival_rate_ns=ns_rate, arrival_rate_ew=ew_rate,
            emergency_prob=em_prob, max_steps=200, seed=2000 + i,
        )
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        avg_queues.append(env.total_waiting_time / 200)
        emg_waits.append(env.total_emergency_waits)
        throughputs.append(env.total_cars_passed)
    return {
        "reward_mean": np.mean(rewards), "reward_std": np.std(rewards),
        "avg_queue": np.mean(avg_queues),
        "emg_wait": np.mean(emg_waits),
        "throughput": np.mean(throughputs),
        "rewards": rewards,
    }


def train_learning_curve(ns_rate, ew_rate, em_prob, max_episodes=10000,
                         checkpoints=None):
    """
    Train Q-learning and evaluate at regular checkpoints to build a
    learning curve.  Returns (checkpoint_episodes, checkpoint_rewards).
    """
    if checkpoints is None:
        checkpoints = list(range(500, max_episodes + 1, 500))

    q_agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3, seed=42)
    rng = np.random.default_rng(1042)

    curve_x, curve_y = [], []
    ep = 0

    for target in checkpoints:
        while ep < target:
            frac = ep / max(max_episodes - 1, 1)
            q_agent.epsilon = 0.3 + (0.01 - 0.3) * frac
            ep_seed = int(rng.integers(0, 100_000))
            env = TrafficEnvironment(
                arrival_rate_ns=ns_rate, arrival_rate_ew=ew_rate,
                emergency_prob=em_prob, max_steps=200, seed=ep_seed,
            )
            s = compress_state(env.reset())
            done = False
            while not done:
                a = q_agent.choose_action(s)
                raw_ns, r, done = env.step(a)
                ns = compress_state(raw_ns)
                q_agent.update(s, a, r, ns)
                s = ns
            ep += 1

        # Evaluate at this checkpoint
        q_agent_copy_eps = q_agent.epsilon
        q_agent.epsilon = 0.0
        hybrid = HybridAgent(q_agent)
        result = evaluate_agent(hybrid, ns_rate, ew_rate, em_prob, n_eval=100)
        curve_x.append(target)
        curve_y.append(result["reward_mean"])
        q_agent.epsilon = q_agent_copy_eps

    return curve_x, curve_y


# ── Plotting ───────────────────────────────────────────────────────────

AGENT_COLORS = {
    "Always Keep": "#888888",
    "Random": "#cc8844",
    "Fixed Timer": "#dd6666",
    "Queue Threshold": "#5599dd",
    "Hybrid Q-Learn": "#44bb66",
}

def plot_metric_bars(all_results, metric, ylabel, title, filename):
    """Grouped bar chart: one group per scenario, one bar per agent."""
    scenarios = list(all_results.keys())
    agents = list(next(iter(all_results.values())).keys())
    n_agents = len(agents)
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, agent_name in enumerate(agents):
        vals = [all_results[sc][agent_name][metric] for sc in scenarios]
        offset = (i - n_agents / 2 + 0.5) * width
        color = AGENT_COLORS.get(agent_name, "#999999")
        ax.bar(x + offset, vals, width * 0.9, label=agent_name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"    Saved {filename}")


def plot_learning_curve(curve_x, curve_y, qt_reward, filename):
    """Learning curve: episode reward over training."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve_x, curve_y, "o-", color="#44bb66", markersize=3,
            label="Hybrid Q-Learn")
    ax.axhline(y=qt_reward, color="#5599dd", linestyle="--", linewidth=1.5,
               label=f"Queue Threshold ({qt_reward:.0f})")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Learning Curve — Hybrid Q-Learn vs Queue Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"    Saved {filename}")


def plot_reward_distribution(all_results, filename):
    """Box plot of per-episode rewards across scenarios for top agents."""
    agents_to_show = ["Queue Threshold", "Hybrid Q-Learn"]
    scenarios = list(all_results.keys())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 4), sharey=False)
    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for ag in agents_to_show:
            data.append(all_results[sc][ag]["rewards"])
            labels.append(ag.replace(" ", "\n"))
            colors.append(AGENT_COLORS[ag])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(sc, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Reward Distribution: Q-Learn vs Queue Threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"    Saved {filename}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Train Hybrid agents (one per scenario) ──
    print("=" * 75)
    print("  TRAINING Hybrid Q-Learning agents (one per scenario)")
    print("=" * 75)
    trained_agents = {}
    for sc_name, params in SCENARIOS.items():
        short = sc_name.replace("\n", " ")
        print(f"  Training for {short}...", end=" ", flush=True)
        agent = train_hybrid(
            arrival_rate_ns=params["ns"],
            arrival_rate_ew=params["ew"],
            emergency_prob=params["em"],
            n_episodes=10000,
        )
        trained_agents[sc_name] = agent
        print("done.", flush=True)
    print()

    # ── 2. Evaluate all agents across all scenarios ──
    print("=" * 75)
    print("  EVALUATING all agents across all scenarios")
    print("=" * 75)

    all_results = {}  # scenario -> agent_name -> metrics

    for sc_name, params in SCENARIOS.items():
        short = sc_name.replace("\n", " ")
        print(f"\n  Scenario: {short}")
        agents = {
            "Always Keep": AlwaysKeepAgent(),
            "Random": RandomAgent(),
            "Fixed Timer": FixedTimerAgent(switch_interval=10),
            "Queue Threshold": QueueThresholdAgent(),
            "Hybrid Q-Learn": trained_agents[sc_name],
        }
        sc_results = {}
        for agent_name, agent in agents.items():
            metrics = evaluate_agent(agent, params["ns"], params["ew"], params["em"])
            sc_results[agent_name] = metrics
            print(f"    {agent_name:>20s}: reward={metrics['reward_mean']:8.1f}  "
                  f"avg_q={metrics['avg_queue']:5.1f}  "
                  f"emg={metrics['emg_wait']:5.1f}  "
                  f"cars={metrics['throughput']:5.1f}", flush=True)
        all_results[sc_name] = sc_results
    print()

    # ── 3. Print summary table ──
    print("=" * 75)
    print("  SUMMARY TABLE")
    print("=" * 75)
    agents_list = ["Always Keep", "Random", "Fixed Timer",
                   "Queue Threshold", "Hybrid Q-Learn"]
    header = f"  {'Scenario':>18s}"
    for a in agents_list:
        header += f" | {a:>16s}"
    print(header)
    print("  " + "-" * (20 + 19 * len(agents_list)))
    for sc_name in SCENARIOS:
        short = sc_name.replace("\n", " ")
        row = f"  {short:>18s}"
        best_reward = max(all_results[sc_name][a]["reward_mean"] for a in agents_list)
        for a in agents_list:
            r = all_results[sc_name][a]["reward_mean"]
            marker = " *" if r == best_reward else "  "
            row += f" | {r:>14.1f}{marker}"
        print(row)
    print()
    print("  * = best agent for that scenario")
    print()

    # ── 4. Generate plots ──
    print("=" * 75)
    print("  GENERATING PLOTS")
    print("=" * 75)

    plot_metric_bars(all_results, "reward_mean", "Mean Episode Reward",
                     "Mean Reward Across Scenarios", "reward_comparison.png")

    plot_metric_bars(all_results, "avg_queue", "Average Queue Length",
                     "Average Queue Length Across Scenarios", "queue_comparison.png")

    plot_metric_bars(all_results, "emg_wait", "Emergency Wait Steps",
                     "Emergency Vehicle Wait Time Across Scenarios",
                     "emergency_comparison.png")

    plot_metric_bars(all_results, "throughput", "Cars Passed",
                     "Throughput (Cars Cleared) Across Scenarios",
                     "throughput_comparison.png")

    plot_reward_distribution(all_results, "reward_distribution.png")

    # ── 5. Learning curve (balanced scenario) ──
    print("\n  Generating learning curve (this takes a moment)...", flush=True)
    curve_x, curve_y = train_learning_curve(0.4, 0.4, 0.05, max_episodes=10000)
    qt_reward = all_results["Balanced\n(0.4 / 0.4)"]["Queue Threshold"]["reward_mean"]
    plot_learning_curve(curve_x, curve_y, qt_reward, "learning_curve.png")

    print(f"\n  All results saved to {RESULTS_DIR}/")
    print("=" * 75)


if __name__ == "__main__":
    main()
