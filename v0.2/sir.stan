functions {
      array[] real sir(real t, array[] real y, array[] real theta, array[] real x_r, array[] int x_i) {
        real S = y[1];
        real I = y[2];
        real R = y[3];
        real N = S + I + R;
        real beta = theta[1];
        real gamma = theta[2];
        real dS_dt = -beta * S * I / N;
        real dI_dt =  beta * S * I / N - gamma * I;
        real dR_dt =  gamma * I;
        return {dS_dt, dI_dt, dR_dt};
      }
    }
        data {
      int<lower=1> n_days;
      array[3] real y0;
      real t0;
      array[n_days] real ts;
      int<lower=1> n_obs;
      array[n_obs] int obs_days;
      array[n_obs] int I_obs;
      int n_x_r;
      array[n_x_r] real x_r;
      int n_x_i;
      array[n_x_i] int x_i;
    }
    parameters {
      real<lower=0> beta;
      real<lower=0> gamma;
      real<lower=0> phi;
    }
        model {
      array[n_days, 3] real y_hat;
      beta ~ normal(0.5, 0.5);
      gamma ~ normal(0.1, 0.1);
      phi ~ normal(0, 1);

      y_hat = integrate_ode_rk45(sir, y0, t0, ts, {beta, gamma}, x_r, x_i);

      for (i in 1:n_obs) {
        I_obs[i] ~ neg_binomial_2_log(log(y_hat[obs_days[i], 2]), phi);
      }
    }
        generated quantities {
        real R0 = beta / gamma;
        array[n_days] real R_t;
        array[n_days, 3] real y_pred;
        y_pred = integrate_ode_rk45(sir, y0, t0, ts, {beta, gamma}, x_r, x_i);

        for (t in 1:n_days) {
            real S_t = y_pred[t, 1];
            real N_t = y_pred[t, 1] + y_pred[t, 2] + y_pred[t, 3];
            R_t[t] = R0 * S_t / N_t;
        }
    }