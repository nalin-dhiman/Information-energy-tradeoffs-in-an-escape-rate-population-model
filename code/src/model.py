import numpy as np

class PopulationModel:
    def __init__(self, cfg):
        self.tau_m = cfg['model'].get('tau_m', 0.02)
        self.tau_a = cfg['model'].get('tau_a', 0.1)
        self.sigma = cfg['model'].get('sigma', 0.5)
        self.kappa = cfg['model'].get('kappa', 1.0)
        
        self.theta0 = cfg['hazard'].get('theta0', 1.0)
        self.thetaV = cfg['hazard'].get('thetaV', 1.0)
        self.thetaa = cfg['hazard'].get('thetaa', 0.0) 
        
        self.thetaVV = cfg['hazard'].get('thetaVV', 0.0)
        self.thetaaa = cfg['hazard'].get('thetaaa', 0.0)
        self.thetaVa = cfg['hazard'].get('thetaVa', 0.0)
        
        self.N = cfg['simulation'].get('N', 1000)
        self.dt = cfg['simulation'].get('dt', 0.001)
        
    def step(self, V, A, S_t, dt):
      
        noise = np.random.randn(self.N) / np.sqrt(dt) * self.sigma
        dV = (-V + self.kappa * S_t + noise) / self.tau_m
        V_new = V + dV * dt
        
        
        
        log_lambda = (self.theta0 + 
                      self.thetaV * V_new + 
                      self.thetaa * A +
                      self.thetaVV * (V_new**2) +
                      self.thetaaa * (A**2) +
                      self.thetaVa * (V_new * A))
        lambd = np.exp(log_lambda)
        
       
        p_spike = 1.0 - np.exp(-lambd * dt)
        spikes = np.random.rand(self.N) < p_spike
        
        
        
        V_final = np.where(spikes, 0.0, V_new)
        
        return V_final, spikes

    def run_batch(self, S, dt):
       
        n_steps = len(S)
        V = np.zeros(self.N)
        A = np.zeros(self.N) 
        
        spike_history = []
        pop_rate = np.zeros(n_steps)
        
        A_pop = 0.0
        
        for i in range(n_steps):
            V, spikes = self.step(V, A_pop, S[i], dt)
            
           
            spike_history.append(spikes)
            
            spike_count = np.sum(spikes)
            dA = (-A_pop + spike_count / (self.N * dt)) / self.tau_a
            A_pop += dA * dt
            
            pop_rate[i] = A_pop
            
        return pop_rate, np.array(spike_history)
