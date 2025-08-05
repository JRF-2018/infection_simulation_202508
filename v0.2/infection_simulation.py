import csv
import random
import numpy as np
from enum import Enum
from collections import defaultdict
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# --- Enums and Data Classes ---

class InfectionStatus(Enum):
    SUSCEPTIBLE = 'S'
    INFECTED = 'I'
    RECOVERED = 'R'
    DECEASED = 'D'

class Person:
    """Represents an individual in the simulation."""
    def __init__(self):
        self.id = None
        self.sex = None
        self.age = None
        self.family = None
        self.organization = None # Work or school class
        self.status = InfectionStatus.SUSCEPTIBLE
        self.is_isolated = False
        self.infection_hour = -1 # Hour of infection
        self.recovery_hour = -1
        self.schedule = {} # Hourly schedule
        self.current_place = None # Current location object

class Family:
    """Represents a household."""
    def __init__(self):
        self.id = None
        self.members = []
        self.internal_infection_prob = 0.05 # High probability for household contact
        self.community_infection_prob = 0.0 # Assume no community infection at home

class Organization:
    """Represents a place where people gather, e.g., school class, company."""
    def __init__(self, org_id, org_type, internal_infection_prob=0.01, community_infection_prob=0.0):
        self.id = org_id
        self.type = org_type
        self.members = []
        self.internal_infection_prob = internal_infection_prob
        self.community_infection_prob = community_infection_prob

# A dictionary to hold all places that are not households
PLACES = {}

# --- Data Loading and Initialization ---

def load_population(file_path):
    """Loads population data from CSV and creates Person and Family objects."""
    persons = {}
    families = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            family_id, _, _, _, person_id, sex, age = row
            if family_id not in families:
                family = Family()
                family.id = family_id
                families[family_id] = family
            else:
                family = families[family_id]

            person = Person()
            person.id = person_id
            person.sex = sex
            person.age = float(age)
            person.family = family
            person.current_place = family # People start at home
            persons[person.id] = person
            family.members.append(person)

    return list(persons.values()), list(families.values())

def assign_organizations(all_persons, avg_company_size=10, max_class_size=30):
    """Assigns individuals to organizations like school classes and companies."""
    # 1. Assign students to classes
    students = [p for p in all_persons if 6 <= p.age < 18]
    num_classes = (len(students) + max_class_size - 1) // max_class_size
    for i in range(num_classes):
        class_id = f"CLASS_{i}"
        # School classes: high internal infection, low community infection
        school_class = Organization(class_id, 'school_class', internal_infection_prob=0.03, community_infection_prob=0.0001)
        PLACES[class_id] = school_class

    for i, student in enumerate(students):
        class_index = i // max_class_size
        class_id = f"CLASS_{class_index}"
        student.organization = PLACES[class_id]
        PLACES[class_id].members.append(student)

    # 2. Assign workers to companies
    workers = [p for p in all_persons if 18 <= p.age < 65 and p.sex == 'M'] # Assume male workers for now
    worker_idx = 0
    company_id_counter = 0
    while worker_idx < len(workers):
        size = np.random.poisson(avg_company_size)
        if size == 0: continue
        company_id = f"COMP_{company_id_counter}"
        # Companies: lower internal infection, some community infection
        company = Organization(company_id, 'company', internal_infection_prob=0.01, community_infection_prob=0.0002)
        PLACES[company_id] = company
        company_id_counter += 1
        for _ in range(size):
            if worker_idx < len(workers):
                worker = workers[worker_idx]
                worker.organization = company
                company.members.append(worker)
                worker_idx += 1
            else: break

def assign_schedules(all_persons):
    """Assigns daily (hourly) schedules to each person."""
    # Public places with different infection characteristics
    # Train: low internal (transient population), medium community
    PLACES['train'] = Organization('train', 'public_transport', internal_infection_prob=0.005, community_infection_prob=0.0005)
    # Downtown: low internal (dispersed), high community
    PLACES['downtown'] = Organization('downtown', 'entertainment', internal_infection_prob=0.002, community_infection_prob=0.001)

    for p in all_persons:
        # Default schedule: stay home
        p.schedule = {h: p.family for h in range(24)}

        if p.organization:
            if p.organization.type == 'school_class':
                # Student schedule
                for h in range(8, 16): p.schedule[h] = p.organization # At school
            elif p.organization.type == 'company':
                # Worker schedule
                p.schedule[7] = PLACES['train'] # Commute
                for h in range(8, 18): p.schedule[h] = p.organization # At work
                p.schedule[18] = PLACES['train'] # Commute back
                if random.random() < 0.3: # 30% go downtown
                    for h in range(19, 22): p.schedule[h] = PLACES['downtown']

# --- Simulation Logic ---

class Simulation:
    def __init__(self, persons, families, params):
        self.persons = persons
        self.families = families
        self.params = params
        self.current_hour = 0
        self.history = []

    def start(self, initial_infected_count):
        """Infects a random number of people to start the simulation."""
        infected_candidates = self.persons
        initial_infected = random.sample(infected_candidates, initial_infected_count)
        self.log_daily_history() # Log initial state at day 0
        for person in initial_infected:
            person.status = InfectionStatus.INFECTED
            person.infection_hour = 0
            person.recovery_hour = self.params['recovery_period_hours']

    def update_locations(self):
        """Update the location of each person based on their schedule."""
        hour_of_day = self.current_hour % 24
        for place in PLACES.values():
            if place.type in ['public_transport', 'entertainment']:
                place.members = []

        for p in self.persons:
            p.current_place = p.family if p.is_isolated else p.schedule.get(hour_of_day, p.family)
            if not isinstance(p.current_place, Family):
                 p.current_place.members.append(p)

    def run_hourly_step(self):
        """Runs one hour of the simulation."""
        if (self.current_hour > 0 and self.current_hour % 24 == 0):
            self.log_daily_history()

        self.update_locations()
        newly_infected = set()
        all_places = list(self.families) + list(PLACES.values())

        # Calculate overall community infection pressure for this hour
        total_non_isolated_infected = sum(1 for p in self.persons if p.status == InfectionStatus.INFECTED and not p.is_isolated)
        community_infection_pressure = total_non_isolated_infected / len(self.persons)

        for place in all_places:
            if not hasattr(place, 'members') or not place.members: continue

            # 1. Internal Infection Probability
            infectious_members = [p for p in place.members if p.status == InfectionStatus.INFECTED]
            prob_internal = 0.0
            if infectious_members:
                place_internal_prob = getattr(place, 'internal_infection_prob', 0.0)
                if place_internal_prob > 0:
                    prob_internal = 1 - (1 - place_internal_prob) ** len(infectious_members)

            # 2. Community (External) Infection Probability for this specific place
            place_community_prob_base = getattr(place, 'community_infection_prob', 0.0)
            prob_community = place_community_prob_base * community_infection_pressure

            if prob_internal == 0.0 and prob_community == 0.0:
                continue

            for person in place.members:
                if person.status == InfectionStatus.SUSCEPTIBLE:
                    # Probability of NOT getting infected from either source
                    prob_no_infection = (1 - prob_internal) * (1 - prob_community)
                    if random.random() > prob_no_infection:
                        newly_infected.add(person)

        # Update status for newly infected
        for person in newly_infected:
            person.status = InfectionStatus.INFECTED
            person.infection_hour = self.current_hour
            person.recovery_hour = self.current_hour + self.params['recovery_period_hours']

        # Recovery, Isolation, and Status Update Phase
        for person in self.persons:
            if person.status == InfectionStatus.INFECTED:
                if self.current_hour >= person.recovery_hour:
                    person.status = InfectionStatus.DECEASED if random.random() < self.params['death_prob'] else InfectionStatus.RECOVERED
                elif not person.is_isolated and (self.current_hour - person.infection_hour) >= self.params['hours_to_isolate']:
                    if random.random() < self.params['isolation_prob']:
                        person.is_isolated = True
        self.current_hour += 1

    def log_daily_history(self):
        """Logs the summary of the current day."""
        counts = defaultdict(int)
        for p in self.persons:
            counts[p.status] += 1
        self.history.append({
            'day': self.current_hour // 24,
            'S': counts[InfectionStatus.SUSCEPTIBLE],
            'I': counts[InfectionStatus.INFECTED],
            'R': counts[InfectionStatus.RECOVERED],
            'D': counts[InfectionStatus.DECEASED],
        })

# --- Analysis and Visualization ---

def plot_results(history, filename="simulation_plot.png"):
    """Plots the simulation results (S, I, R counts over time)."""
    if not history:
        print("History is empty. Cannot plot results.")
        return
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(12, 8))
    plt.plot(history_df['day'], history_df['S'], label='Susceptible')
    plt.plot(history_df['day'], history_df['I'], label='Infected')
    plt.plot(history_df['day'], history_df['R'], label='Recovered')
    plt.plot(history_df['day'], history_df['D'], label='Deceased')
    plt.xlabel('Day')
    plt.ylabel('Number of People')
    plt.title('Hourly Infection Simulation Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Simulation plot saved to {filename}")

def analyze_with_stan(history, population_size, initial_infected_count):
    """
    Analyzes the simulation results with a SIR model using CmdStanPy.
    """
    print("\nStarting Stan analysis...")
    history_df = pd.DataFrame(history)
    I_obs = history_df['I'].values
    if I_obs[0] == 0: I_obs[0] = initial_infected_count

    stan_data = {
        'n_days': len(history),
        'y0': [population_size - initial_infected_count, initial_infected_count, 0],
        't0': 0,
        'ts': list(range(1, len(history) + 1)),
        'n_obs': len(I_obs),
        'obs_days': list(range(1, len(I_obs) + 1)),
        'I_obs': I_obs.astype(int),
        'n_x_r': 0, 'x_r': [], 'n_x_i': 0, 'x_i': []
    }

    try:
        model = CmdStanModel(stan_file='sir.stan')
        fit = model.sample(data=stan_data, chains=4, iter_sampling=1000, show_progress=True)

        print("Creating Stan trace plot...")
        az.plot_trace(az.from_cmdstanpy(posterior=fit), var_names=['beta', 'gamma', 'R0'])
        plt.tight_layout()
        plt.savefig("stan_trace.png")
        plt.close()
        print("Stan trace plot saved to stan_trace.png")

        print("Creating R(t) plot...")
        R_t_mean = fit.stan_variable('R_t').mean(axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['day'], R_t_mean, label='Estimated R(t)')
        plt.axhline(y=1, color='r', linestyle='--', label='R(t) = 1')
        plt.xlabel('Day'); plt.ylabel('R(t)'); plt.title('Estimated R(t)')
        plt.legend(); plt.grid(True)
        plt.savefig("rt_plot.png")
        plt.close()
        print("R(t) plot saved to rt_plot.png")

        print("\n--- Stan Analysis Summary ---")
        summary_df = fit.summary()
        # Display Mean, Median (50%), StdDev, and 90% credible interval (5% to 95%)
        print(summary_df.loc[['beta', 'gamma', 'R0'], ['Mean', 'StdDev', '5%', '50%', '95%']])
        print("---------------------------\n")

    except Exception as e:
        print(f"An error occurred during Stan analysis: {e}")

# --- Main Execution ---

if __name__ == '__main__':
    SIM_DAYS = 100
    POPULATION_FILE = 'simple_synthetic_population/population-1000.csv'
    INITIAL_INFECTED = 5

    sim_params = {
        'recovery_period_hours': 14 * 24,
        'death_prob': 0.01,
        'isolation_prob': 0.8,
        'hours_to_isolate': 3 * 24,
    }

    print("Loading population...")
    persons, families = load_population(POPULATION_FILE)
    print("Assigning organizations and schedules...")
    assign_organizations(persons)
    assign_schedules(persons)

    print("Starting simulation...")
    simulation = Simulation(persons, families, sim_params)
    simulation.start(initial_infected_count=INITIAL_INFECTED)

    total_hours = SIM_DAYS * 24
    for hour in range(total_hours):
        if (hour % (24 * 10) == 0):
            print(f"Simulating... Hour {hour}/{total_hours} (Day {hour // 24})")
        simulation.run_hourly_step()
    simulation.log_daily_history()

    print("Simulation finished.")
    final_counts = simulation.history[-1]
    print(f"--- Simulation Report (After {SIM_DAYS} days) ---")
    print(f"Susceptible: {final_counts['S']}")
    print(f"Infected:    {final_counts['I']}")
    print(f"Recovered:   {final_counts['R']}")
    print(f"Deceased:    {final_counts['D']}")

    plot_results(simulation.history)
    analyze_with_stan(simulation.history, len(persons), INITIAL_INFECTED)