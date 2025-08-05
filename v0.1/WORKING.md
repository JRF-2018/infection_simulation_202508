# Working Log

## 2025-08-05

*   Cloned the repository `JRF-2018/simple_synthetic_population`.
*   Created initial files: `WORKING.md`, `REPORT.md`, `infection_simulation.py`.
*   Implemented CSV data loading to create `Person` and `Family` objects.
*   Extended `Person` and `Family` classes and added a new `Organization` class.
*   Implemented `assign_organizations` function to assign individuals to schools or companies.
*   Used `numpy` to assign company sizes based on a Poisson distribution.
*   Implemented the core simulation logic with a `Simulation` class.
*   The simulation now models infection spread through organizations, families, and the community.
*   Ran a 100-day simulation and confirmed the model works as expected.
*   Created `sir.stan` to define a SIR model for analysis.
*   Implemented `analyze_with_stan` function in `infection_simulation.py` to run the Stan model using `cmdstanpy`.
*   The analysis estimates R0 and R(t) from the simulation data.
*   Generated `stan_trace.png` and `rt_plot.png` to visualize the results of the Stan analysis.
*   Implemented self-isolation logic in `infection_simulation.py`.
*   Updated `REPORT.md` to reflect the new self-isolation feature.
*   First run with isolation resulted in a very small outbreak, causing Stan model convergence issues.
*   Adjusted simulation parameters (isolation probability, community infection rate) to generate a more significant outbreak.
*   Final run executed successfully, producing a meaningful infection curve and allowing the Stan model to converge properly. Project complete.