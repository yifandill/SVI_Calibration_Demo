import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint
from config import SuppressOutput


class SVICalibrator:
    def __init__(self, epsilon=1e-3, verbose=False):
        """
        Initialize SVI Calibrator
        
        Parameters:
        epsilon : float - Small constant for constraints
        verbose : bool - If True, print detailed output during calibration
        """
        self.epsilon = epsilon
        self.calibration_results = []
        self.verbose = verbose
        
    def svi_raw(self, k, params):
        """
        Raw SVI parameterization of total implied variance
        
        Parameters:
        k : array-like - Log-moneyness k = log(K/F_T)
        params : tuple - SVI parameters (a, b, rho, m, sigma)
        """
        a, b, rho, m, sigma = params
        x = k - m
        return a + b * (rho * x + np.sqrt(x**2 + sigma**2))
    
    def g_function(self, k, params):
        """Calculate g function for butterfly arbitrage condition"""
        a, b, rho, m, sigma = params
        x = k - m
        R = np.sqrt(x**2 + sigma**2)
        W = self.svi_raw(k, params)  # omega(k)
        U = b * (rho + x / R)   # omega'(k)
        
        # Second derivative
        omega_pp = b * sigma**2 / R**3
        
        # g function calculation
        g = (1 - k * U / (2 * W))**2 - (U**2 / 4) * (1/W + 1/4) + omega_pp/2
        
        return g
    
    def objective(self, params, k, market_variance):
        """Sum of squared errors between model and market"""
        model_variance = self.svi_raw(k, params)
        return np.sum((model_variance - market_variance)**2)
    
    def butterfly_arbitrage_constraint(self, params, k):
        """Butterfly arbitrage constraint: g(k) > epsilon"""
        # Check g function on a fine grid including input points
        min_k, max_k = np.min(k), np.max(k)
        grid_k = np.linspace(min_k - 0.5, max_k + 0.5, 100)
        g_values = self.g_function(grid_k, params)
        return g_values - self.epsilon
    
    def calendar_spread_constraint(self, params_current, params_prev, k_current, k_prev=None):
        """
        Calendar spread arbitrage constraint: w(k, T_current) > w(k, T_prev) + epsilon
        
        Handles different k grids between maturities by creating a common evaluation grid
        """
        # Create a common grid covering both k ranges
        if k_prev is None:
            k_prev = k_current
            
        k_min = min(np.min(k_current), np.min(k_prev))
        k_max = max(np.max(k_current), np.max(k_prev))
        
        # Extend the grid slightly
        k_min -= 0.5
        k_max += 0.5
        
        # Dense common grid for evaluating both functions
        grid_k = np.linspace(k_min, k_max, 200)
        
        # Evaluate both variance functions on the common grid
        w_current = self.svi_raw(grid_k, params_current)
        w_prev = self.svi_raw(grid_k, params_prev)
        
        return w_current - w_prev - self.epsilon
    
    def get_parameter_bounds(self, k, market_variance):
        """Get parameter bounds according to paper constraints"""
        a_min, a_max = 1e-5, np.max(market_variance)
        b_min, b_max = 0.001, 1.0
        rho_min, rho_max = -0.999, 0.999  # Avoid exactly -1 or 1
        m_min, m_max = 2 * np.min(k), 2 * np.max(k)
        sigma_min, sigma_max = 0.01, 1.0
        
        return Bounds(
            [a_min, b_min, rho_min, m_min, sigma_min],
            [a_max, b_max, rho_max, m_max, sigma_max]
        )
    
    def get_initial_guess(self, market_variance, prev_params=None):
        """Get initial parameter guess"""
        if prev_params is not None:
            return prev_params
        
        # Default initial guess as per the paper
        a = 0.5 * np.min(market_variance)
        b = 0.1
        rho = -0.5
        m = 0.1
        sigma = 0.1
        
        return (a, b, rho, m, sigma)
        
    def calibrate_single(self, k, market_variance, initial_guess=None):
        """Calibrate SVI model for a single maturity"""
        if initial_guess is None:
            initial_guess = self.get_initial_guess(market_variance)
        
        bounds = self.get_parameter_bounds(k, market_variance)
        
        # Nonlinear constraint for butterfly arbitrage
        nlc = NonlinearConstraint(
            lambda params: self.butterfly_arbitrage_constraint(params, k),
            lb=0, ub=np.inf
        )
        
        # Minimize options with display turned off
        options = {'ftol': 1e-9, 'maxiter': 1000}
        if self.verbose:
            options['disp'] = True
        
        # Suppress output during optimization if not verbose
        with SuppressOutput(suppress_stdout=not self.verbose):
            # Optimize using SLSQP
            result = minimize(
                self.objective,
                initial_guess,
                args=(k, market_variance),
                method='SLSQP',
                bounds=bounds,
                constraints=[nlc],
                options=options
            )
        
        if not result.success and self.verbose:
            print(f"Warning: Optimization did not converge. Status: {result.status}, Message: {result.message}")
        
        return result.x
    
    def calibrate_multi(self, k_list, market_variance_list, initial_guess_list=None):
        """
        Calibrate SVI model for multiple maturities with calendar spread constraints
        
        Parameters:
        k_list : list of arrays - Log-moneyness for each maturity (can be different lengths/values)
        market_variance_list : list of arrays - Market implied variances for each maturity
        initial_guess_list : list of tuples - Initial guesses for each maturity
        
        Returns:
        list of parameter tuples - Calibrated parameters for each maturity
        """
        n_maturities = len(k_list)
        self.calibration_results = []
        
        if initial_guess_list is None:
            initial_guess_list = [None] * n_maturities
        elif len(initial_guess_list) < n_maturities:
            initial_guess_list = initial_guess_list + [None] * (n_maturities - len(initial_guess_list))
        
        # Calibrate first maturity (no calendar spread constraint)
        if self.verbose:
            print(f"Calibrating maturity 1 of {n_maturities}...")
            
        first_params = self.calibrate_single(
            k_list[0], 
            market_variance_list[0], 
            initial_guess_list[0]
        )
        self.calibration_results.append(first_params)
        
        # Calibrate subsequent maturities with calendar spread constraints
        for i in range(1, n_maturities):
            if self.verbose:
                print(f"Calibrating maturity {i+1} of {n_maturities}...")
                
            init_guess = initial_guess_list[i] if initial_guess_list[i] is not None else self.calibration_results[-1]
            
            bounds = self.get_parameter_bounds(k_list[i], market_variance_list[i])
            
            # Butterfly arbitrage constraint
            butterfly_nlc = NonlinearConstraint(
                lambda params: self.butterfly_arbitrage_constraint(params, k_list[i]),
                lb=0, ub=np.inf
            )
            
            # Calendar spread constraint with previous maturity
            prev_params = self.calibration_results[-1]
            cal_spread_nlc = NonlinearConstraint(
                lambda params: self.calendar_spread_constraint(params, prev_params, k_list[i], k_list[i-1]),
                lb=0, ub=np.inf
            )
            
            # Options without display
            options = {'ftol': 1e-9, 'maxiter': 1000}
            if self.verbose:
                options['disp'] = True
            
            # Suppress output during optimization if not verbose
            with SuppressOutput(suppress_stdout=not self.verbose):
                # Optimize using SLSQP
                result = minimize(
                    self.objective,
                    init_guess,
                    args=(k_list[i], market_variance_list[i]),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[butterfly_nlc, cal_spread_nlc],
                    options=options
                )
            
            if not result.success and self.verbose:
                print(f"Warning: Optimization for maturity {i} did not converge. Status: {result.status}")
                print(f"Message: {result.message}")
            
            self.calibration_results.append(result.x)
        
        return self.calibration_results

    def robust_calibration(self, k, market_variance, max_attempts=3):
        """
        More robust calibration that tries multiple initial guesses and relaxed constraints
        
        Parameters:
        k : array-like - Log-moneyness k = log(K/F_T)
        market_variance : array-like - Market implied variances
        max_attempts : int - Maximum number of calibration attempts with different guesses
        
        Returns:
        tuple - Calibrated SVI parameters (a, b, rho, m, sigma)
        """
        # First try: standard calibration
        try:
            params = self.calibrate_single(k, market_variance)
            return params
        except Exception as e:
            if self.verbose:
                print(f"First calibration attempt failed: {e}")
        
        # Second try: different initial guesses
        initial_guesses = [
            (0.005, 0.1, -0.5, 0.0, 0.1),
            (0.001, 0.5, 0.0, 0.0, 0.2),
            (0.0005, 0.2, -0.7, np.mean(k), 0.05)
        ]
        
        for i, guess in enumerate(initial_guesses[:max_attempts-1]):
            try:
                if self.verbose:
                    print(f"Attempting calibration with alternative initial guess {i+1}...")
                params = self.calibrate_single(k, market_variance, guess)
                return params
            except Exception as e:
                if self.verbose:
                    print(f"Calibration attempt {i+2} failed: {e}")
        
        # Last resort: simple parameter estimation
        if self.verbose:
            print("Using simple parameter estimation...")
            
        a_simple = np.min(market_variance) * 0.9
        max_diff = np.max(market_variance) - np.min(market_variance)
        b_simple = max_diff / 0.5
        rho_simple = -0.5
        min_var_idx = np.argmin(market_variance)
        m_simple = k[min_var_idx]
        sigma_simple = 0.1
        
        simple_params = (a_simple, b_simple, rho_simple, m_simple, sigma_simple)
        return simple_params
    
    def robust_calibrate_multi(self, k_list, market_variance_list, max_attempts=3):
        """
        Robustly calibrate SVI model for multiple maturities with calendar spread constraints
        
        Parameters:
        k_list : list of arrays - Log-moneyness for each maturity
        market_variance_list : list of arrays - Market implied variances for each maturity
        max_attempts : int - Maximum number of calibration attempts for each maturity
        
        Returns:
        list of parameter tuples - Calibrated parameters for each maturity
        """
        n_maturities = len(k_list)
        self.calibration_results = []
        
        # Calibrate first maturity (no calendar spread constraint)
        if self.verbose:
            print(f"Calibrating maturity 1 of {n_maturities}...")
            
        first_params = self.robust_calibration(k_list[0], market_variance_list[0], max_attempts)
        self.calibration_results.append(first_params)
        
        # Calibrate subsequent maturities with calendar spread constraints
        for i in range(1, n_maturities):
            if self.verbose:
                print(f"Calibrating maturity {i+1} of {n_maturities}...")
            
            # First try standard multi-slice calibration
            try:
                # Use previous params as initial guess
                prev_params = self.calibration_results[-1]
                
                bounds = self.get_parameter_bounds(k_list[i], market_variance_list[i])
                
                # Butterfly arbitrage constraint
                butterfly_nlc = NonlinearConstraint(
                    lambda params: self.butterfly_arbitrage_constraint(params, k_list[i]),
                    lb=0, ub=np.inf
                )
                
                # Calendar spread constraint with previous maturity
                cal_spread_nlc = NonlinearConstraint(
                    lambda params: self.calendar_spread_constraint(params, prev_params, k_list[i], k_list[i-1]),
                    lb=0, ub=np.inf
                )
                
                # Options without display
                options = {'ftol': 1e-9, 'maxiter': 1000}
                if self.verbose:
                    options['disp'] = True
                
                # Suppress output during optimization if not verbose
                with SuppressOutput(suppress_stdout=not self.verbose):
                    # Optimize using SLSQP
                    result = minimize(
                        self.objective,
                        prev_params,
                        args=(k_list[i], market_variance_list[i]),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=[butterfly_nlc, cal_spread_nlc],
                        options=options
                    )
                
                if result.success:
                    params = result.x
                    self.calibration_results.append(params)
                    continue
                else:
                    raise Exception(f"Optimization failed: {result.message}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Standard multi-slice calibration failed: {e}")
                    print("Trying robust calibration without calendar spread constraint...")
                
                params = self.robust_calibration(k_list[i], market_variance_list[i], max_attempts)
                
                # Check if the calendar spread constraint is violated
                grid_k = np.linspace(np.min(k_list[i-1])-0.5, np.max(k_list[i-1])+0.5, 100)
                prev_params = self.calibration_results[-1]
                w_current = self.svi_raw(grid_k, params)
                w_prev = self.svi_raw(grid_k, prev_params)
                min_diff = np.min(w_current - w_prev)
                
                if min_diff < self.epsilon:
                    if self.verbose:
                        print(f"Warning: Calendar spread arbitrage detected, min diff = {min_diff:.6f}")
                        print("Adjusting parameters to avoid calendar spread arbitrage...")
                    
                    # Simple adjustment: increase 'a' parameter to shift the curve up
                    a, b, rho, m, sigma = params
                    a_adjusted = a + abs(min_diff) + 2*self.epsilon
                    params = (a_adjusted, b, rho, m, sigma)
                    
                    # Verify adjustment worked
                    w_current = self.svi_raw(grid_k, params)
                    min_diff = np.min(w_current - w_prev)
                    if self.verbose:
                        print(f"After adjustment, min diff = {min_diff:.6f}")
                
                self.calibration_results.append(params)
        
        return self.calibration_results
    
    def get_parameters_dataframe(self, maturities=None):
        """
        Convert calibrated parameters to a pandas DataFrame
        
        Parameters:
        maturities : list of float - Time to maturity in years (must match length of calibration_results)
        
        Returns:
        pandas.DataFrame - DataFrame containing all calibrated parameters indexed by maturity
        """
        import pandas as pd
        
        if not self.calibration_results:
            raise ValueError("No calibration results available")
        
        if maturities is None:
            # Use generic indices if no maturities provided
            maturities = [f"T{i+1}" for i in range(len(self.calibration_results))]
        
        if len(maturities) != len(self.calibration_results):
            raise ValueError(f"Number of maturities ({len(maturities)}) doesn't match number of calibration results ({len(self.calibration_results)})")
        
        # Create dictionary of parameters
        params_dict = {}
        for i, params in enumerate(self.calibration_results):
            a, b, rho, m, sigma = params
            
            # Format maturity as string with appropriate precision if it's a float
            if isinstance(maturities[i], float):
                mat_key = f"{maturities[i]:.8f}"
            else:
                mat_key = str(maturities[i])
                
            params_dict[mat_key] = {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma
            }
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(params_dict, orient='index')
        df.index.name = 'Maturity'
        
        return df
