import numpy as np

class PopulationModel:
    def __init__(self, cfg):
        self.tau_m = cfg['model'].get('tau_m', 0.02)
        self.tau_a = cfg['model'].get('tau_a', 0.1)
        self.sigma = cfg['model'].get('sigma', 0.5)
        self.kappa = cfg['model'].get('kappa', 1.0) # Input gain
        
        self.theta0 = cfg['hazard'].get('theta0', 1.0)
        self.thetaV = cfg['hazard'].get('thetaV', 1.0)
        self.thetaa = cfg['hazard'].get('thetaa', 0.0) # For coupling
        
        self.thetaVV = cfg['hazard'].get('thetaVV', 0.0)
        self.thetaaa = cfg['hazard'].get('thetaaa', 0.0)
        self.thetaVa = cfg['hazard'].get('thetaVa', 0.0)
        
        self.N = cfg['simulation'].get('N', 1000)
        self.dt = cfg['simulation'].get('dt', 0.001)
        
    def step(self, V, A, S_t, dt):
        """
        Evolve V and generate spikes.
        V: (N,) membrane potentials
        A: (N,) filtered activity (or scalar population A if mean field)
        S_t: current stimulus value
        """
        # Dynamics of V: dV/dt = (-V + kappa*S + noise)/tau_m
        noise = np.random.randn(self.N) / np.sqrt(dt) * self.sigma
        # We can treat S_t as scalar or vector if there's spatial structure. Assuming scalar S(t).
        dV = (-V + self.kappa * S_t + noise) / self.tau_m
        V_new = V + dV * dt
        
        # Hazard
        # lambda = exp(theta0 + thetaV * V + thetaa * A + quadratic terms)
        
        log_lambda = (self.theta0 + 
                      self.thetaV * V_new + 
                      self.thetaa * A +
                      self.thetaVV * (V_new**2) +
                      self.thetaaa * (A**2) +
                      self.thetaVa * (V_new * A))
        lambd = np.exp(log_lambda)
        
        # Spikes
        # Prob = 1 - exp(-lambda * dt) approx lambda * dt
        p_spike = 1.0 - np.exp(-lambd * dt)
        spikes = np.random.rand(self.N) < p_spike
        
        # Reset? Renewal model usually resets V, but hazard model might not.
        # "Hazard-based renewal model" implies reset.
        # If it's a GLM, usually V is not reset but spike history filter is added.
        # "Renewal" strictly means inter-spike intervals are independent.
        # If V(t) is driven by S(t) and minimal self-history, a reset makes it renewal-like.
        # Standard assumption: V -> V_reset after spike.
        # But if it's "hazard-based", maybe it's just inhomogeneous Poisson?
        # User says "hazard-based renewal model".
        # I'll implement a reset to 0.
        
        V_final = np.where(spikes, 0.0, V_new)
        
        return V_final, spikes

    def run_batch(self, S, dt):
        """
        Run for full duration of S.
        """
        n_steps = len(S)
        V = np.zeros(self.N)
        A = np.zeros(self.N) # Local activity filter? Or global?
        # If theta_a is used, we need A.
        # Let's track global A for coupling, but here it is 0.
        
        spike_history = []
        # Store population rate?
        pop_rate = np.zeros(n_steps)
        
        A_pop = 0.0
        
        for i in range(n_steps):
            V, spikes = self.step(V, A_pop, S[i], dt)
            
            # Save sparse spikes for analysis?
            # Creating list of lists or masked array
            spike_history.append(spikes) # heavy if detailed
            
            # Update A_pop
            spike_count = np.sum(spikes)
            dA = (-A_pop + spike_count / (self.N * dt)) / self.tau_a
            A_pop += dA * dt
            
            pop_rate[i] = A_pop
            
        return pop_rate, np.array(spike_history)
