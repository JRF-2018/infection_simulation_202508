import csv
import random
import numpy as np
from enum import Enum
from collections import defaultdict
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

class InfectionStatus(Enum):
    SUSCEPTIBLE = 'S'
    INFECTED = 'I'
    RECOVERED = 'R'
    DECEASED = 'D'

class Person:
    def __init__ (self):
        self.id = None
        self.sex = None
        self.age = None
        self.family = None
        self.status = InfectionStatus.SUSCEPTIBLE
        self.is_isolated = False
        self.organization = None
        self.mode = 'on'
        self.infection_day = -1 # Day of infection

class Family:
    def __init__ (self):
        self.id = None
        self.code = None
        self.num = None
        self.master = None
        self.spouse = None
        self.parents = []
        self.children = []
        self.others = []
        self.members = []

class Organization:
    def __init__(self, org_id, org_type):
        self.id = org_id
        self.type = org_type
        self.members = []
        self.internal_infection_prob = 0.0

def load_population(file_path):
    persons = {}
    families = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            family_id, family_code, family_num, relationship, person_id, sex, age = row
            if family_id not in families:
                family = Family()
                family.id = family_id
                family.code = int(family_code)
                family.num = int(family_num)
                families[family_id] = family
            else:
                family = families[family_id]
            person = Person()
            person.id = person_id
            person.sex = sex
            person.age = float(age)
            person.family = family
            persons[person.id] = person
            family.members.append(person)
            if relationship == 'M': family.master = person
            elif relationship == 'S': family.spouse = person
            elif relationship == 'P': family.parents.append(person)
            elif relationship == 'C': family.children.append(person)
            elif relationship == 'O': family.others.append(person)
    return list(persons.values()), list(families.values())

def assign_organizations(all_persons, avg_company_size=10):
    organizations = []
    school = Organization("S001", 'school')
    school.internal_infection_prob = 0.05 # Higher infection rate in schools
    organizations.append(school)
    students = [p for p in all_persons if 6 <= p.age < 18]
    for student in students:
        student.organization = school
        school.members.append(student)

    workers = [p for p in all_persons if 18 <= p.age < 65]
    worker_idx = 0
    company_id_counter = 0
    while worker_idx < len(workers):
        size = np.random.poisson(avg_company_size)
        if size == 0: continue
        company_id = f"C{str(company_id_counter).zfill(4)}"
        company = Organization(company_id, 'company')
        company.internal_infection_prob = 0.01 # Lower infection rate in companies
        organizations.append(company)
        company_id_counter += 1
        for _ in range(size):
            if worker_idx < len(workers):
                worker = workers[worker_idx]
                worker.organization = company
                company.members.append(worker)
                worker_idx += 1
            else: break
    return organizations

class Simulation:
    def __init__(self, persons, families, organizations, params):
        self.persons = persons
        self.families = families
        self.organizations = organizations
        self.params = params
        self.current_day = 0
        self.history = []

    def start(self, initial_infected_count):
        # Initial infection
        infected_candidates = [p for p in self.persons if not p.is_isolated]
        initial_infected = random.sample(infected_candidates, initial_infected_count)
        for person in initial_infected:
            person.status = InfectionStatus.INFECTED
            person.infection_day = 0

    def run_step(self):
        self.current_day += 1
        newly_infected = []

        # --- Self-Isolation Phase ---
        for person in self.persons:
            if person.status == InfectionStatus.INFECTED and not person.is_isolated:
                if (self.current_day - person.infection_day) >= self.params['days_to_isolate']:
                    if random.random() < self.params['isolation_prob']:
                        person.is_isolated = True

        # --- Infection Phase ---
        # 1. Organization
        for org in self.organizations:
            infected_count = sum(1 for p in org.members if p.status == InfectionStatus.INFECTED and not p.is_isolated)
            if infected_count == 0: continue
            
            prob = 1 - (1 - org.internal_infection_prob) ** infected_count
            for person in org.members:
                if person.status == InfectionStatus.SUSCEPTIBLE and random.random() < prob:
                    newly_infected.append(person)

        # 2. Family
        for family in self.families:
            infected_count = sum(1 for p in family.members if p.status == InfectionStatus.INFECTED)
            if infected_count == 0: continue

            prob = 1 - (1 - self.params['family_infection_prob']) ** infected_count
            for person in family.members:
                if person.status == InfectionStatus.SUSCEPTIBLE and random.random() < prob:
                    if person not in newly_infected:
                        newly_infected.append(person)
        
        # 3. Community
        total_infected_count = sum(1 for p in self.persons if p.status == InfectionStatus.INFECTED and not p.is_isolated)
        community_prob = 1 - (1 - self.params['community_infection_prob']) ** (total_infected_count / len(self.persons))
        for person in self.persons:
            if person.status == InfectionStatus.SUSCEPTIBLE and random.random() < community_prob:
                 if person not in newly_infected:
                    newly_infected.append(person)

        # Update status for newly infected
        for person in newly_infected:
            person.status = InfectionStatus.INFECTED
            person.infection_day = self.current_day

        # --- Recovery/Status Update Phase ---
        for person in self.persons:
            if person.status == InfectionStatus.INFECTED:
                if (self.current_day - person.infection_day) >= self.params['recovery_period']:
                    if random.random() < self.params['death_prob']:
                        person.status = InfectionStatus.DECEASED
                    else:
                        person.status = InfectionStatus.RECOVERED
        
        self.log_history()

    def log_history(self):
        counts = defaultdict(int)
        isolated_count = 0
        for p in self.persons:
            counts[p.status] += 1
            if p.is_isolated:
                isolated_count += 1
        self.history.append({
            'day': self.current_day,
            'S': counts[InfectionStatus.SUSCEPTIBLE],
            'I': counts[InfectionStatus.INFECTED],
            'R': counts[InfectionStatus.RECOVERED],
            'D': counts[InfectionStatus.DECEASED],
            'Isolated': isolated_count,
        })

def analyze_with_stan(history, population_size):
    """
    Analyzes the simulation results with a SIR model using CmdStanPy.
    """
    # Prepare data for Stan
    history_df = pd.DataFrame(history)
    I_obs = history_df['I'].values
    n_obs = len(I_obs)
    obs_days = list(range(1, n_obs + 1))

    stan_data = {
        'n_days': len(history),
        'y0': [population_size - 1, 1, 0],
        't0': 0,
        'ts': list(range(1, len(history) + 1)),
        'n_obs': n_obs,
        'obs_days': obs_days,
        'I_obs': I_obs,
        'n_x_r': 0,
        'x_r': [],
        'n_x_i': 0,
        'x_i': []
    }

    # Build and run the model
    model = CmdStanModel(stan_file='sir.stan')
    fit = model.sample(data=stan_data, chains=4, iter_sampling=1000)


    # Visualize and print results
    az_data = az.from_cmdstanpy(posterior=fit)
    az.plot_trace(az_data, var_names=['beta', 'gamma', 'R0'])
    plt.tight_layout()
    plt.savefig("stan_trace.png")
    plt.close()


    # R(t)
    R_t_mean = fit.stan_variable('R_t').mean(axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['day'], R_t_mean, label='Estimated R(t)')
    plt.axhline(y=1, color='r', linestyle='--', label='R(t) = 1')
    plt.xlabel('Day')
    plt.ylabel('R(t)')
    plt.title('Estimated Effective Reproduction Number R(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig("rt_plot.png")
    plt.close()

if __name__ == '__main__':
    persons, families = load_population('simple_synthetic_population/population-1000.csv')
    organizations = assign_organizations(persons)

    sim_params = {
        'family_infection_prob': 0.02,
        'community_infection_prob': 0.002,
        'recovery_period': 14, # days
        'death_prob': 0.01,
        'isolation_prob': 0.5, # Probability of self-isolation after symptoms
        'days_to_isolate': 3, # Days from infection to isolation
    }

    simulation = Simulation(persons, families, organizations, sim_params)
    simulation.start(initial_infected_count=5)

    num_days = 100
    for day in range(num_days):
        simulation.run_step()

    # Print final results
    final_counts = simulation.history[-1]
    print(f"--- Simulation Report (After {num_days} days) ---")
    print(f"Susceptible: {final_counts['S']}")
    print(f"Infected:    {final_counts['I']}")
    print(f"Recovered:   {final_counts['R']}")
    print(f"Deceased:    {final_counts['D']}")
    print(f"Isolated:    {final_counts['Isolated']}")

    analyze_with_stan(simulation.history, len(persons))