import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from config import SuppressOutput


class SVICalibrator:
    """
    SVI Calibrator using differential evolution algorithm for optimization.
    
    This implementation focuses on stable parameter estimation and 
    includes arbitrage constraints.
    """
    
    def __init__(self, epsilon=1e-6, verbose=False, popsize=15, mutation=(0.5, 1.0), recombination=0.7, max_iterations=1000):
        """
        Initialize SVI Calibrator
        
        Parameters:
        ----------
        epsilon : float
            Small constant for constraints
        verbose : bool
            Whether to display optimization details
        popsize : int
            Population size for differential evolution
        mutation : tuple
            Mutation parameters for differential evolution
        recombination : float
            Recombination probability for differential evolution
        max_iterations : int
            Maximum number of iterations for optimization
        """
        self.epsilon = epsilon
        self.verbose = verbose
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.max_iterations = max_iterations
        self.calibration_results = []
    
    def svi_raw(self, k, params):
        """
        Raw SVI parameterization of total implied variance
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        params : tuple
            SVI parameters (a, b, rho, m, sigma)
            
        Returns:
        -------
        array-like
            Total variance
        """
        a, b, rho, m, sigma = params
        k = np.asarray(k)
        x = k - m
        return a + b * (rho * x + np.sqrt(x**2 + sigma**2))
    
    def g_function(self, k, params):
        """
        Calculate g function for butterfly arbitrage condition
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        params : tuple
            SVI parameters (a, b, rho, m, sigma)
            
        Returns:
        -------
        array-like
            g function values
        """
        a, b, rho, m, sigma = params
        k = np.asarray(k)
        x = k - m
        R = np.sqrt(x**2 + sigma**2)
        W = self.svi_raw(k, params)  # omega(k)
        U = b * (rho + x / R)   # omega'(k)
        
        # Second derivative
        omega_pp = b * sigma**2 / R**3
        
        # g function calculation
        g = (1 - k * U / (2 * W))**2 - (U**2 / 4) * (1/W + 1/4) + omega_pp/2
        
        return g
    
    def butterfly_arbitrage_constraint(self, params, k):
        """
        Butterfly arbitrage constraint penalty function
        
        Parameters:
        ----------
        params : tuple
            SVI parameters (a, b, rho, m, sigma)
        k : array-like
            Log-moneyness k = log(K/F_T)
            
        Returns:
        -------
        float
            Penalty value (0 if no arbitrage, >0 if arbitrage detected)
        """
        k_grid = np.linspace(np.min(k) - 0.5, np.max(k) + 0.5, 100)
        g_values = self.g_function(k_grid, params)
        min_g = np.min(g_values)
        
        # Return penalty if arbitrage constraint violated
        return max(0, -min_g + self.epsilon)
    
    def calendar_spread_constraint(self, params_current, params_prev, k_current, k_prev=None):
        """
        Calendar spread arbitrage constraint penalty function
        
        Parameters:
        ----------
        params_current : tuple
            SVI parameters for current maturity
        params_prev : tuple
            SVI parameters for previous maturity
        k_current : array-like
            Log-moneyness for current maturity
        k_prev : array-like, optional
            Log-moneyness for previous maturity
            
        Returns:
        -------
        float
            Penalty value (0 if no arbitrage, >0 if arbitrage detected)
        """
        if k_prev is None:
            k_prev = k_current
        
        # Create common grid
        k_min = min(np.min(k_current), np.min(k_prev))
        k_max = max(np.max(k_current), np.max(k_prev))
        
        # Extend the grid
        k_min -= 0.5
        k_max += 0.5
        
        grid_k = np.linspace(k_min, k_max, 100)
        
        # Evaluate variance functions
        w_current = self.svi_raw(grid_k, params_current)
        w_prev = self.svi_raw(grid_k, params_prev)
        
        # Check calendar spread constraint
        diff = w_current - w_prev
        min_diff = np.min(diff)
        
        # Return penalty if calendar spread constraint violated
        return max(0, -min_diff + self.epsilon)
    
    def objective_function(self, params, k, market_variance, include_butterfly=True):
        """
        Objective function for SVI calibration with optional constraints
        
        Parameters:
        ----------
        params : array-like
            SVI parameters (a, b, rho, m, sigma)
        k : array-like
            Log-moneyness k = log(K/F_T)
        market_variance : array-like
            Market implied total variance
        include_butterfly : bool
            Whether to include butterfly arbitrage constraint
            
        Returns:
        -------
        float
            Error value including penalties
        """
        # Unpack parameters
        a, b, rho, m, sigma = params
        
        # Calculate model variance
        model_variance = self.svi_raw(k, params)
        
        # Error scaling by moneyness (give more weight to ATM points)
        error_scale = np.exp(-np.abs(k) * 0.5)
        
        # Calculate weighted MSE
        errors = (model_variance - market_variance)**2 * error_scale
        mse = np.mean(errors)
        
        # Add butterfly arbitrage penalty if requested
        penalty = 0
        if include_butterfly:
            butterfly_penalty = self.butterfly_arbitrage_constraint(params, k)
            penalty += 1000 * butterfly_penalty  # High weight to enforce constraint
        
        return mse + penalty
    
    def get_parameter_bounds(self):
        """
        Get parameter bounds for optimization
        
        Returns:
        -------
        list
            List of (min, max) bounds for each parameter
        """
        return [
            (1e-8, 0.1),      # a: small but positive
            (1e-4, 2.0),      # b: positive but not too large
            (-0.999, 0.999),  # rho: between -1 and 1, avoiding extremes
            (-2.0, 2.0),      # m: reasonable range for log-moneyness shift
            (1e-4, 1.0)       # sigma: positive but not too large
        ]
    
    def get_initial_guess(self, k, market_variance):
        """
        Generate initial guess for SVI parameters
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        market_variance : array-like
            Market implied total variance
            
        Returns:
        -------
        tuple
            Initial guess for SVI parameters
        """
        # Simple heuristic for initial parameters
        a = np.min(market_variance) * 0.9  # slightly below min variance
        b = (np.max(market_variance) - np.min(market_variance)) / 0.5  # slope estimate
        rho = -0.5  # moderate negative correlation
        
        # Find k value with minimum variance
        min_var_idx = np.argmin(market_variance)
        m = k[min_var_idx]  # shift parameter at minimum variance point
        
        sigma = 0.1  # moderate curvature
        
        return (a, b, rho, m, sigma)
    
    def calibrate_single(self, k, market_variance, initial_guess=None):
        """
        Calibrate SVI model for a single maturity using differential evolution
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        market_variance : array-like
            Market implied total variance
        initial_guess : tuple, optional
            Initial guess for SVI parameters
            
        Returns:
        -------
        tuple
            Calibrated SVI parameters (a, b, rho, m, sigma)
        """
        # Convert inputs to numpy arrays
        k = np.asarray(k)
        market_variance = np.asarray(market_variance)
        
        # Get initial guess if not provided
        if initial_guess is None:
            initial_guess = self.get_initial_guess(k, market_variance)
        
        # Set optimization bounds
        bounds = self.get_parameter_bounds()
        
        # Optimization options
        options = {
            'popsize': self.popsize,
            'mutation': self.mutation,
            'recombination': self.recombination,
            'maxiter': self.max_iterations
        }
        
        # Suppress output during optimization if not verbose
        with SuppressOutput(suppress_stdout=not self.verbose):
            # Run differential evolution
            result = differential_evolution(
                lambda params: self.objective_function(params, k, market_variance, True),
                bounds=bounds,
                x0=initial_guess,
                strategy='best1bin',
                polish=True,
                **options
            )
        
        if not result.success and self.verbose:
            print(f"Warning: Optimization did not converge. Status: {result.status}, Message: {result.message}")
        
        return tuple(result.x)
    
    def robust_calibration(self, k, market_variance, max_attempts=3):
        """
        More robust calibration with multiple attempts and different initial guesses
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        market_variance : array-like
            Market implied total variance
        max_attempts : int
            Maximum number of calibration attempts
            
        Returns:
        -------
        tuple
            Calibrated SVI parameters (a, b, rho, m, sigma)
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
            (0.001, 0.5, 0.0, np.mean(k), 0.2),
            (0.0005, 0.2, -0.7, np.median(k), 0.05)
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
        
        # Simple estimation based on data shape
        a_simple = np.min(market_variance) * 0.9
        max_diff = np.max(market_variance) - np.min(market_variance)
        b_simple = max_diff / 0.5
        rho_simple = -0.5
        min_var_idx = np.argmin(market_variance)
        m_simple = k[min_var_idx]
        sigma_simple = 0.1
        
        simple_params = (a_simple, b_simple, rho_simple, m_simple, sigma_simple)
        return simple_params
    
    def calibrate_multi(self, k_list, market_variance_list, initial_guess_list=None):
        """
        Calibrate SVI model for multiple maturities with calendar spread constraints
        
        Parameters:
        ----------
        k_list : list of arrays
            Log-moneyness for each maturity
        market_variance_list : list of arrays
            Market implied variances for each maturity
        initial_guess_list : list of tuples, optional
            Initial guesses for each maturity
            
        Returns:
        -------
        list
            List of calibrated SVI parameters for each maturity
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
            
            prev_params = self.calibration_results[-1]
            k_current = k_list[i]
            market_variance = market_variance_list[i]
            
            init_guess = initial_guess_list[i] if initial_guess_list[i] is not None else prev_params
            
            # Create objective function with calendar spread constraint
            def objective_with_constraints(params):
                # Basic fitting error
                basic_error = self.objective_function(params, k_current, market_variance, True)
                
                # Calendar spread constraint
                cal_spread_penalty = self.calendar_spread_constraint(
                    params, prev_params, k_current, k_list[i-1]
                )
                
                # Add high penalty for calendar spread violation
                return basic_error + 1000 * cal_spread_penalty
            
            # Set bounds
            bounds = self.get_parameter_bounds()
            
            # Optimization options
            options = {
                'popsize': self.popsize,
                'mutation': self.mutation,
                'recombination': self.recombination,
                'maxiter': self.max_iterations
            }
            
            # Suppress output during optimization if not verbose
            with SuppressOutput(suppress_stdout=not self.verbose):
                # Run differential evolution
                try:
                    result = differential_evolution(
                        objective_with_constraints,
                        bounds=bounds,
                        x0=init_guess,
                        strategy='best1bin',
                        polish=True,
                        **options
                    )
                    
                    if result.success:
                        params = tuple(result.x)
                        self.calibration_results.append(params)
                    else:
                        raise Exception(f"Optimization did not converge: {result.message}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"Calibration with calendar spread constraint failed: {e}")
                        print("Falling back to unconstrained calibration...")
                    
                    # Fall back to simple calibration without calendar spread
                    params = self.calibrate_single(k_current, market_variance, init_guess)
                    
                    # Check if calendar spread constraint is violated
                    penalty = self.calendar_spread_constraint(
                        params, prev_params, k_current, k_list[i-1]
                    )
                    
                    if penalty > 0 and self.verbose:
                        print(f"Warning: Calendar spread arbitrage detected, penalty = {penalty:.6f}")
                        print("Adjusting parameters to avoid calendar spread arbitrage...")
                    
                    # Simple adjustment if violated: increase 'a' parameter
                    if penalty > 0:
                        a, b, rho, m, sigma = params
                        a_adjusted = a + penalty + self.epsilon
                        params = (a_adjusted, b, rho, m, sigma)
                    
                    self.calibration_results.append(params)
        
        return self.calibration_results
    
    def robust_calibrate_multi(self, k_list, market_variance_list, max_attempts=3):
        """
        Robust calibration for multiple maturities with fallback options
        
        Parameters:
        ----------
        k_list : list of arrays
            Log-moneyness for each maturity
        market_variance_list : list of arrays
            Market implied variances for each maturity
        max_attempts : int
            Maximum number of calibration attempts per maturity
            
        Returns:
        -------
        list
            List of calibrated SVI parameters for each maturity
        """
        n_maturities = len(k_list)
        self.calibration_results = []
        
        # Calibrate first maturity with robust approach
        if self.verbose:
            print(f"Calibrating maturity 1 of {n_maturities}...")
            
        first_params = self.robust_calibration(
            k_list[0], market_variance_list[0], max_attempts
        )
        self.calibration_results.append(first_params)
        
        # Calibrate subsequent maturities
        for i in range(1, n_maturities):
            if self.verbose:
                print(f"Calibrating maturity {i+1} of {n_maturities}...")
            
            try:
                # First try calibration with calendar spread constraint
                prev_params = self.calibration_results[-1]
                k_current = k_list[i]
                market_variance = market_variance_list[i]
                
                # Create objective function with calendar spread constraint
                def objective_with_constraints(params):
                    # Basic fitting error
                    basic_error = self.objective_function(params, k_current, market_variance, True)
                    
                    # Calendar spread constraint
                    cal_spread_penalty = self.calendar_spread_constraint(
                        params, prev_params, k_current, k_list[i-1]
                    )
                    
                    # Add high penalty for calendar spread violation
                    return basic_error + 1000 * cal_spread_penalty
                
                # Set bounds and options
                bounds = self.get_parameter_bounds()
                options = {
                    'popsize': self.popsize,
                    'mutation': self.mutation,
                    'recombination': self.recombination,
                    'maxiter': self.max_iterations
                }
                
                # Suppress output during optimization if not verbose
                with SuppressOutput(suppress_stdout=not self.verbose):
                    result = differential_evolution(
                        objective_with_constraints,
                        bounds=bounds,
                        x0=prev_params,
                        strategy='best1bin',
                        polish=True,
                        **options
                    )
                
                if result.success:
                    params = tuple(result.x)
                    self.calibration_results.append(params)
                else:
                    raise Exception(f"Optimization did not converge: {result.message}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Standard multi-slice calibration failed: {e}")
                    print("Trying robust calibration without calendar spread constraint...")
                
                # Fall back to robust calibration without calendar spread
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
        ----------
        maturities : list, optional
            Time to maturity in years
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame containing calibrated parameters
        """
        if not self.calibration_results:
            raise ValueError("No calibration results available")
        
        if maturities is None:
            # Use generic indices if no maturities provided
            maturities = [f"T{i+1}" for i in range(len(self.calibration_results))]
        
        if len(maturities) != len(self.calibration_results):
            raise ValueError(
                f"Number of maturities ({len(maturities)}) doesn't match number of calibration results ({len(self.calibration_results)})"
            )
        
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
