from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import simpy


# ----------------------------
# Shared utility functionality
# ----------------------------


class QueueLengthTracker:
    """Tracks queue length as a step function over simulation time."""

    def __init__(self) -> None:
        self.times: List[float] = [0.0]
        self.lengths: List[int] = [0]

    def record(self, time_point: float, queue_length: int) -> None:
        if time_point < self.times[-1]:
            raise ValueError("Queue events must be recorded in chronological order.")

        if np.isclose(time_point, self.times[-1]):
            self.lengths[-1] = queue_length
        else:
            self.times.append(time_point)
            self.lengths.append(queue_length)

    def clipped_series(self, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        clipped_times: List[float] = []
        clipped_lengths: List[int] = []

        for time_point, length in zip(self.times, self.lengths):
            if time_point <= horizon:
                clipped_times.append(time_point)
                clipped_lengths.append(length)
            else:
                break

        if not clipped_times:
            clipped_times = [0.0, horizon]
            clipped_lengths = [0, 0]
        elif clipped_times[-1] < horizon:
            clipped_times.append(horizon)
            clipped_lengths.append(clipped_lengths[-1])

        return np.array(clipped_times), np.array(clipped_lengths)


def time_weighted_average(times: Sequence[float], values: Sequence[int], horizon: float) -> float:
    if len(times) != len(values):
        raise ValueError("times and values must have equal length.")
    if not times:
        return 0.0

    area = 0.0
    for idx in range(len(times) - 1):
        start = times[idx]
        end = min(times[idx + 1], horizon)
        if end > start:
            area += values[idx] * (end - start)

    if times[-1] < horizon:
        area += values[-1] * (horizon - times[-1])

    return area / horizon if horizon > 0 else 0.0


def overlap_with_horizon(interval_start: float, interval_end: float, horizon: float) -> float:
    return max(0.0, min(interval_end, horizon) - max(interval_start, 0.0))


def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ----------------------------------------------------------
# Problem 1: Nigerian Commercial Bank (M/M/1 and M/M/2)
# ----------------------------------------------------------


@dataclass
class MMQueueResult:
    waiting_times: np.ndarray
    avg_waiting_time: float
    utilization: float
    avg_queue_length: float
    queue_times: np.ndarray
    queue_lengths: np.ndarray
    served_count: int


def simulate_mm_queue(
    arrival_rate_per_hour: float,
    service_rate_per_hour: float,
    num_servers: int,
    duration_hours: float,
    seed: int,
) -> MMQueueResult:
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=num_servers)

    arrival_rng = np.random.default_rng(seed)
    service_rng = np.random.default_rng(seed + 10_000)

    queue_tracker = QueueLengthTracker()
    waiting_times: List[float] = []
    service_intervals: List[Tuple[float, float]] = []

    waiting_count = 0

    def customer_process(customer_id: int) -> simpy.events.Process:
        del customer_id  # ID included for readability/extension even if unused in this assignment.

        nonlocal waiting_count
        arrival_time = env.now

        # Service time is sampled at arrival to keep comparisons reproducible across scenarios.
        service_time = float(service_rng.exponential(scale=1 / service_rate_per_hour))

        waiting_count += 1
        queue_tracker.record(env.now, waiting_count)

        with server.request() as request:
            yield request

            waiting_count -= 1
            queue_tracker.record(env.now, waiting_count)

            service_start = env.now
            waiting_times.append(service_start - arrival_time)

            yield env.timeout(service_time)
            service_end = env.now
            service_intervals.append((service_start, service_end))

    def arrival_process() -> simpy.events.Process:
        customer_id = 0
        while env.now < duration_hours:
            inter_arrival = float(arrival_rng.exponential(scale=1 / arrival_rate_per_hour))
            yield env.timeout(inter_arrival)
            if env.now > duration_hours:
                break

            customer_id += 1
            env.process(customer_process(customer_id))

    env.process(arrival_process())
    env.run()

    queue_times, queue_lengths = queue_tracker.clipped_series(duration_hours)

    busy_time_in_horizon = sum(
        overlap_with_horizon(start, end, duration_hours) for start, end in service_intervals
    )
    utilization = busy_time_in_horizon / (num_servers * duration_hours)

    avg_queue_length = time_weighted_average(
        times=queue_times.tolist(),
        values=queue_lengths.tolist(),
        horizon=duration_hours,
    )

    waiting_np = np.array(waiting_times, dtype=float)

    return MMQueueResult(
        waiting_times=waiting_np,
        avg_waiting_time=float(np.mean(waiting_np)) if waiting_np.size > 0 else 0.0,
        utilization=utilization,
        avg_queue_length=avg_queue_length,
        queue_times=queue_times,
        queue_lengths=queue_lengths,
        served_count=int(waiting_np.size),
    )


def solve_problem_1(figures_dir: Path) -> Dict[str, MMQueueResult]:
    print("\n" + "=" * 90)
    print("PROBLEM 1 - NIGERIAN COMMERCIAL BANK QUEUE (M/M/1 vs M/M/2)")
    print("=" * 90)

    arrival_rate = 15.0  # customers/hour
    service_rate = 20.0  # customers/hour per teller
    duration = 8.0  # hours
    seed = 42

    result_one_teller = simulate_mm_queue(
        arrival_rate_per_hour=arrival_rate,
        service_rate_per_hour=service_rate,
        num_servers=1,
        duration_hours=duration,
        seed=seed,
    )

    result_two_tellers = simulate_mm_queue(
        arrival_rate_per_hour=arrival_rate,
        service_rate_per_hour=service_rate,
        num_servers=2,
        duration_hours=duration,
        seed=seed,
    )

    # Display side-by-side metrics.
    print(f"Simulation duration: {duration:.1f} hours")
    print(f"Arrival rate (lambda): {arrival_rate:.1f} customers/hour")
    print(f"Service rate (mu): {service_rate:.1f} customers/hour per teller")
    print("\nMetric                           1 Teller (M/M/1)         2 Tellers (M/M/2)")
    print("-" * 78)
    print(
        f"Average waiting time (minutes)   {result_one_teller.avg_waiting_time * 60:>10.3f}"
        f"{result_two_tellers.avg_waiting_time * 60:>24.3f}"
    )
    print(
        f"Server utilization (%)           {result_one_teller.utilization * 100:>10.2f}"
        f"{result_two_tellers.utilization * 100:>24.2f}"
    )
    print(
        f"Average queue length             {result_one_teller.avg_queue_length:>10.3f}"
        f"{result_two_tellers.avg_queue_length:>24.3f}"
    )
    print(
        f"Customers served                 {result_one_teller.served_count:>10d}"
        f"{result_two_tellers.served_count:>24d}"
    )

    # Graph: queue length vs time for both scenarios.
    plt.figure(figsize=(10, 5.5))
    plt.step(
        result_one_teller.queue_times,
        result_one_teller.queue_lengths,
        where="post",
        linewidth=1.6,
        label="1 Teller (M/M/1)",
    )
    plt.step(
        result_two_tellers.queue_times,
        result_two_tellers.queue_lengths,
        where="post",
        linewidth=1.6,
        label="2 Tellers (M/M/2)",
    )
    plt.title("Problem 1: Queue Length vs Time (Bank)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Queue Length (waiting customers)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = figures_dir / "problem1_queue_length_vs_time.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Graph saved: {output_path}")

    return {"mm1": result_one_teller, "mm2": result_two_tellers}


# -----------------------------------------------------------------
# Problem 2: Hospital Emergency Room (Priority vs FIFO vs LIFO)
# -----------------------------------------------------------------


@dataclass
class ERRunResult:
    strategy: str
    waits_all: List[float]
    waits_emergency: List[float]
    waits_normal: List[float]


class ERSingleServerSimulation:
    """Single-doctor ER simulation with selectable queue discipline."""

    def __init__(
        self,
        strategy: str,
        arrival_rate_per_hour: float,
        service_rate_per_hour: float,
        emergency_probability: float,
        duration_hours: float,
        seed: int,
    ) -> None:
        self.strategy = strategy.upper()
        self.arrival_rate = arrival_rate_per_hour
        self.service_rate = service_rate_per_hour
        self.emergency_probability = emergency_probability
        self.duration = duration_hours

        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)

        self.queue: List[Dict[str, float | str]] = []
        self.arrivals_finished = False
        self.item_available = self.env.event()

        self.waits_all: List[float] = []
        self.waits_emergency: List[float] = []
        self.waits_normal: List[float] = []

    def select_next_patient_index(self) -> int:
        if self.strategy == "FIFO":
            return 0

        if self.strategy == "LIFO":
            return len(self.queue) - 1

        if self.strategy == "PRIORITY":
            # Emergency patients are always selected first.
            for index, patient in enumerate(self.queue):
                if patient["category"] == "Emergency":
                    return index
            return 0

        raise ValueError(f"Unsupported strategy: {self.strategy}")

    def notify_item_available(self) -> None:
        if not self.item_available.triggered:
            self.item_available.succeed()

    def arrival_process(self) -> simpy.events.Process:
        patient_id = 0

        while self.env.now < self.duration:
            inter_arrival = float(self.rng.exponential(scale=1 / self.arrival_rate))
            yield self.env.timeout(inter_arrival)
            if self.env.now > self.duration:
                break

            patient_id += 1
            category = "Emergency" if self.rng.random() < self.emergency_probability else "Normal"
            service_time = float(self.rng.exponential(scale=1 / self.service_rate))
            self.queue.append(
                {
                    "id": patient_id,
                    "arrival": self.env.now,
                    "category": category,
                    "service_time": service_time,
                }
            )
            self.notify_item_available()

        self.arrivals_finished = True
        self.notify_item_available()

    def doctor_process(self) -> simpy.events.Process:
        while True:
            if not self.queue:
                if self.arrivals_finished:
                    break

                self.item_available = self.env.event()
                yield self.item_available
                continue

            next_index = self.select_next_patient_index()
            patient = self.queue.pop(next_index)

            wait_time = self.env.now - float(patient["arrival"])
            self.waits_all.append(wait_time)

            if patient["category"] == "Emergency":
                self.waits_emergency.append(wait_time)
            else:
                self.waits_normal.append(wait_time)

            yield self.env.timeout(float(patient["service_time"]))

    def run(self) -> ERRunResult:
        self.env.process(self.arrival_process())
        self.env.process(self.doctor_process())
        self.env.run()

        return ERRunResult(
            strategy=self.strategy,
            waits_all=self.waits_all,
            waits_emergency=self.waits_emergency,
            waits_normal=self.waits_normal,
        )


@dataclass
class ERStrategyStats:
    mean_wait_emergency: float
    std_wait_emergency: float
    mean_wait_normal: float
    std_wait_normal: float
    mean_wait_all: float
    std_wait_all: float


def solve_problem_2(figures_dir: Path) -> Dict[str, ERStrategyStats]:
    print("\n" + "=" * 90)
    print("PROBLEM 2 - HOSPITAL ER (PRIORITY vs FIFO vs LIFO)")
    print("=" * 90)

    arrival_rate = 10.0  # patients/hour
    service_rate = 12.0  # patients/hour
    emergency_probability = 0.30
    duration = 8.0  # hours
    replications = 50

    strategies = ["PRIORITY", "FIFO", "LIFO"]

    strategy_stats: Dict[str, ERStrategyStats] = {}
    representative_runs: Dict[str, ERRunResult] = {}

    for strategy_idx, strategy in enumerate(strategies):
        run_avg_emergency: List[float] = []
        run_avg_normal: List[float] = []
        run_avg_all: List[float] = []

        for replication in range(replications):
            seed = 1000 + strategy_idx * 10_000 + replication
            run = ERSingleServerSimulation(
                strategy=strategy,
                arrival_rate_per_hour=arrival_rate,
                service_rate_per_hour=service_rate,
                emergency_probability=emergency_probability,
                duration_hours=duration,
                seed=seed,
            ).run()

            run_avg_emergency.append(mean_or_zero(run.waits_emergency))
            run_avg_normal.append(mean_or_zero(run.waits_normal))
            run_avg_all.append(mean_or_zero(run.waits_all))

            # Keep first run for patient-level distribution plots.
            if replication == 0:
                representative_runs[strategy] = run

        strategy_stats[strategy] = ERStrategyStats(
            mean_wait_emergency=float(np.mean(run_avg_emergency)),
            std_wait_emergency=float(np.std(run_avg_emergency, ddof=1)),
            mean_wait_normal=float(np.mean(run_avg_normal)),
            std_wait_normal=float(np.std(run_avg_normal, ddof=1)),
            mean_wait_all=float(np.mean(run_avg_all)),
            std_wait_all=float(np.std(run_avg_all, ddof=1)),
        )

    print(f"Simulation duration per replication: {duration:.1f} hours")
    print(f"Arrival rate (lambda): {arrival_rate:.1f} patients/hour")
    print(f"Service rate (mu): {service_rate:.1f} patients/hour")
    print(f"Emergency proportion: {emergency_probability * 100:.0f}%")
    print(f"Replications per strategy: {replications}")

    print("\nStrategy    Emergency Wait (min)      Normal Wait (min)         Overall Wait (min)")
    print("-" * 86)
    for strategy in strategies:
        stats = strategy_stats[strategy]
        print(
            f"{strategy:<10}"
            f"{stats.mean_wait_emergency * 60:>8.2f} +/- {stats.std_wait_emergency * 60:<8.2f}"
            f"{stats.mean_wait_normal * 60:>12.2f} +/- {stats.std_wait_normal * 60:<8.2f}"
            f"{stats.mean_wait_all * 60:>12.2f} +/- {stats.std_wait_all * 60:<8.2f}"
        )

    # Graph 1: Grouped bar chart (mean waiting time by category and strategy).
    x = np.arange(len(strategies))
    width = 0.35
    emergency_means = [strategy_stats[s].mean_wait_emergency * 60 for s in strategies]
    normal_means = [strategy_stats[s].mean_wait_normal * 60 for s in strategies]

    plt.figure(figsize=(10, 5.8))
    plt.bar(x - width / 2, emergency_means, width=width, label="Emergency")
    plt.bar(x + width / 2, normal_means, width=width, label="Normal")
    plt.xticks(x, strategies)
    plt.ylabel("Average Waiting Time (minutes)")
    plt.title("Problem 2: Average Waiting Time by Strategy and Patient Type")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path_1 = figures_dir / "problem2_waiting_time_grouped_bar.png"
    plt.savefig(output_path_1, dpi=160)
    plt.close()

    # Graph 2: Patient-level waiting-time distributions for each strategy.
    patient_wait_distributions = [
        np.array(representative_runs[s].waits_all, dtype=float) * 60 for s in strategies
    ]

    plt.figure(figsize=(10, 5.8))
    plt.boxplot(patient_wait_distributions, tick_labels=strategies, showfliers=False)
    plt.ylabel("Patient Waiting Time (minutes)")
    plt.title("Problem 2: Waiting-Time Distribution Across Queue Strategies")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path_2 = figures_dir / "problem2_waiting_time_boxplot.png"
    plt.savefig(output_path_2, dpi=160)
    plt.close()

    print(f"Graphs saved: {output_path_1}")
    print(f"Graphs saved: {output_path_2}")

    return strategy_stats


# -----------------------------------------------------------------------
# Problem 3: Nigerian Fintech App Request Queue with Spike and Scaling
# -----------------------------------------------------------------------


@dataclass
class RequestQueueResult:
    scenario: str
    avg_response_time: float
    avg_wait_time: float
    utilization: float
    avg_queue_size: float
    max_queue_size: int
    served_count: int
    queue_times: np.ndarray
    queue_sizes: np.ndarray
    response_times: np.ndarray
    arrival_times: np.ndarray


class FintechRequestQueueSimulation:
    """Encapsulates request-queue simulation state for Problem 3 scenarios."""

    def __init__(
        self,
        scenario: str,
        arrival_schedule: Sequence[Tuple[float, float, float]],
        service_rate_per_second: float,
        num_servers: int,
        duration_seconds: float,
        seed: int,
    ) -> None:
        self.scenario = scenario
        self.arrival_schedule = arrival_schedule
        self.service_rate = service_rate_per_second
        self.duration = duration_seconds

        self.env = simpy.Environment()
        self.server = simpy.Resource(self.env, capacity=num_servers)

        self.arrival_rng = np.random.default_rng(seed)
        self.service_rng = np.random.default_rng(seed + 25_000)

        self.queue_tracker = QueueLengthTracker()
        self.waiting_count = 0
        self.next_request_id = 0

        self.response_times: List[float] = []
        self.wait_times: List[float] = []
        self.arrival_times: List[float] = []
        self.service_intervals: List[Tuple[float, float]] = []

    def request_process(self, request_id: int) -> simpy.events.Process:
        del request_id

        arrival_time = self.env.now
        self.arrival_times.append(arrival_time)

        # Service time sampled at arrival for reproducibility across scenarios.
        service_time = float(self.service_rng.exponential(scale=1 / self.service_rate))

        self.waiting_count += 1
        self.queue_tracker.record(self.env.now, self.waiting_count)

        with self.server.request() as request:
            yield request

            self.waiting_count -= 1
            self.queue_tracker.record(self.env.now, self.waiting_count)

            service_start = self.env.now
            self.wait_times.append(service_start - arrival_time)

            yield self.env.timeout(service_time)
            service_end = self.env.now

            self.response_times.append(service_end - arrival_time)
            self.service_intervals.append((service_start, service_end))

    def emit_arrivals_in_segment(self, segment_end: float, rate: float) -> simpy.events.Process:
        while self.env.now < segment_end:
            inter_arrival = float(self.arrival_rng.exponential(scale=1 / rate))
            next_arrival_time = self.env.now + inter_arrival

            if next_arrival_time > segment_end:
                yield self.env.timeout(segment_end - self.env.now)
                return

            yield self.env.timeout(inter_arrival)
            self.next_request_id += 1
            self.env.process(self.request_process(self.next_request_id))

    def arrivals_process(self) -> simpy.events.Process:
        for start, end, rate in self.arrival_schedule:
            if start >= self.duration:
                return

            segment_end = min(end, self.duration)

            if self.env.now < start:
                yield self.env.timeout(start - self.env.now)

            if rate <= 0:
                if self.env.now < segment_end:
                    yield self.env.timeout(segment_end - self.env.now)
                continue

            yield self.env.process(self.emit_arrivals_in_segment(segment_end, rate))

    def run(self) -> RequestQueueResult:
        self.env.process(self.arrivals_process())
        self.env.run()

        queue_times, queue_sizes = self.queue_tracker.clipped_series(self.duration)

        busy_time = sum(
            overlap_with_horizon(start, end, self.duration)
            for start, end in self.service_intervals
        )
        utilization = busy_time / (self.server.capacity * self.duration)

        avg_queue_size = time_weighted_average(
            times=queue_times.tolist(),
            values=queue_sizes.tolist(),
            horizon=self.duration,
        )

        response_np = np.array(self.response_times, dtype=float)
        wait_np = np.array(self.wait_times, dtype=float)
        arrival_np = np.array(self.arrival_times, dtype=float)

        return RequestQueueResult(
            scenario=self.scenario,
            avg_response_time=float(np.mean(response_np)) if response_np.size > 0 else 0.0,
            avg_wait_time=float(np.mean(wait_np)) if wait_np.size > 0 else 0.0,
            utilization=utilization,
            avg_queue_size=avg_queue_size,
            max_queue_size=int(np.max(queue_sizes)) if queue_sizes.size > 0 else 0,
            served_count=int(response_np.size),
            queue_times=queue_times,
            queue_sizes=queue_sizes,
            response_times=response_np,
            arrival_times=arrival_np,
        )


def simulate_request_queue(
    scenario: str,
    arrival_schedule: Sequence[Tuple[float, float, float]],
    service_rate_per_second: float,
    num_servers: int,
    duration_seconds: float,
    seed: int,
) -> RequestQueueResult:
    simulation = FintechRequestQueueSimulation(
        scenario=scenario,
        arrival_schedule=arrival_schedule,
        service_rate_per_second=service_rate_per_second,
        num_servers=num_servers,
        duration_seconds=duration_seconds,
        seed=seed,
    )
    return simulation.run()


def solve_problem_3(figures_dir: Path) -> Dict[str, RequestQueueResult]:
    print("\n" + "=" * 90)
    print("PROBLEM 3 - NIGERIAN FINTECH APP REQUEST QUEUE")
    print("=" * 90)

    duration = 60.0  # seconds
    service_rate = 120.0  # requests/second per server

    baseline_schedule = [(0.0, 60.0, 100.0)]
    spike_schedule = [
        (0.0, 20.0, 100.0),
        (20.0, 40.0, 150.0),
        (40.0, 60.0, 100.0),
    ]

    baseline = simulate_request_queue(
        scenario="Baseline (100 rps, 1 server)",
        arrival_schedule=baseline_schedule,
        service_rate_per_second=service_rate,
        num_servers=1,
        duration_seconds=duration,
        seed=900,
    )

    spike = simulate_request_queue(
        scenario="Spike (100->150->100 rps, 1 server)",
        arrival_schedule=spike_schedule,
        service_rate_per_second=service_rate,
        num_servers=1,
        duration_seconds=duration,
        seed=901,
    )

    scaled = simulate_request_queue(
        scenario="Spike (100->150->100 rps, 2 servers)",
        arrival_schedule=spike_schedule,
        service_rate_per_second=service_rate,
        num_servers=2,
        duration_seconds=duration,
        seed=901,
    )

    def window_average_response(result: RequestQueueResult, start: float, end: float) -> float:
        if result.arrival_times.size == 0:
            return 0.0
        mask = (result.arrival_times >= start) & (result.arrival_times < end)
        if not np.any(mask):
            return 0.0
        return float(np.mean(result.response_times[mask]))

    spike_window_start = 20.0
    spike_window_end = 40.0

    spike_window_response_single = window_average_response(spike, spike_window_start, spike_window_end)
    spike_window_response_scaled = window_average_response(scaled, spike_window_start, spike_window_end)

    print(f"Simulation duration: {duration:.0f} seconds")
    print(f"Service rate (mu): {service_rate:.0f} requests/second per server")
    print("\nScenario                              Avg Response (ms)   Avg Queue Size   Max Queue   Utilization (%)")
    print("-" * 103)
    for result in [baseline, spike, scaled]:
        print(
            f"{result.scenario:<36}"
            f"{result.avg_response_time * 1000:>12.3f}"
            f"{result.avg_queue_size:>18.3f}"
            f"{result.max_queue_size:>12d}"
            f"{result.utilization * 100:>16.2f}"
        )

    print(
        "\nSpike-window response time (20s-40s): "
        f"single server = {spike_window_response_single * 1000:.3f} ms, "
        f"two servers = {spike_window_response_scaled * 1000:.3f} ms"
    )

    # Graph 1: queue size over time for all scenarios.
    plt.figure(figsize=(10.5, 6.0))
    plt.step(baseline.queue_times, baseline.queue_sizes, where="post", label=baseline.scenario)
    plt.step(spike.queue_times, spike.queue_sizes, where="post", label=spike.scenario)
    plt.step(scaled.queue_times, scaled.queue_sizes, where="post", label=scaled.scenario)
    plt.title("Problem 3: Queue Size Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Queue Size (waiting requests)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path_1 = figures_dir / "problem3_queue_size_over_time.png"
    plt.savefig(output_path_1, dpi=160)
    plt.close()

    # Graph 2: response time comparison to demonstrate scaling effect.
    labels = ["Baseline\n(1 server)", "Spike\n(1 server)", "Spike\n(2 servers)"]
    response_ms = [
        baseline.avg_response_time * 1000,
        spike.avg_response_time * 1000,
        scaled.avg_response_time * 1000,
    ]

    plt.figure(figsize=(9.5, 5.8))
    bars = plt.bar(labels, response_ms)
    plt.ylabel("Average Response Time (ms)")
    plt.title("Problem 3: Response Time Impact of Spike and Scaling")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, response_ms):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path_2 = figures_dir / "problem3_response_time_comparison.png"
    plt.savefig(output_path_2, dpi=160)
    plt.close()

    print(f"Graph saved: {output_path_1}")
    print(f"Graph saved: {output_path_2}")

    print(
        "Scaling strategy demonstrated: increasing from 1 to 2 servers during high traffic "
        "substantially reduces queue buildup and response time under spike load."
    )

    return {
        "baseline": baseline,
        "spike_single_server": spike,
        "spike_two_servers": scaled,
    }


# ---------
# Main flow
# ---------


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    _ = solve_problem_1(figures_dir)
    _ = solve_problem_2(figures_dir)
    _ = solve_problem_3(figures_dir)

    print("\n" + "=" * 90)
    print("ALL SIMULATIONS COMPLETED")
    print("=" * 90)
    print(f"All generated graphs are in: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
