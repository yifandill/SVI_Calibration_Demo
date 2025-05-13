import numpy as np
import pandas as pd
from scipy import optimize as opt
from config import SuppressOutput


class SVICalibrator:
    """
    SVI Calibrator using Quasi-Explicit method for optimization.
    
    This implementation uses a two-step iterative approach:
    1. Fix m and sigma, solve for a, d, c using linear least squares
    2. Fix a, d, c, optimize m and sigma using Nelder-Mead
    Repeat until convergence
    """
    
    def __init__(self, epsilon=1e-6, verbose=False, maxiter=10, exit_tol=1e-12):
        """
        Initialize SVI Calibrator with Quasi-Explicit method
        
        Parameters:
        ----------
        epsilon : float
            Small constant for constraints
        verbose : bool
            Whether to display optimization details
        maxiter : int
            Maximum number of iterations for the two-step process
        exit_tol : float
            Exit tolerance for convergence
        """
        self.epsilon = epsilon
        self.verbose = verbose
        self.maxiter = maxiter
        self.exit_tol = exit_tol
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
        centered = k - m
        return a + b * (rho * centered + np.sqrt(centered**2 + sigma**2))
    
    def svi_quasi(self, x, a, d, c, m, sigma):
        """
        Quasi-explicit SVI parameterization
        
        Parameters:
        ----------
        x : array-like
            Log-moneyness
        a, d, c, m, sigma : float
            QE parameters
            
        Returns:
        -------
        array-like
            Total variance
        """
        y = (x - m) / sigma
        return a + d * y + c * np.sqrt(y**2 + 1)
    
    def quasi2raw(self, a, d, c, m, sigma):
        """
        Convert quasi-explicit parameters to raw SVI parameters
        
        Parameters:
        ----------
        a, d, c, m, sigma : float
            QE parameters
            
        Returns:
        -------
        tuple
            Raw SVI parameters (a, b, rho, m, sigma)
        """
        # Ensure we don't divide by zero
        c = max(c, 1e-10)
        b = c / sigma
        rho = d / c
        return a, b, rho, m, sigma
    
    def raw2quasi(self, a, b, rho, m, sigma):
        """
        Convert raw SVI parameters to quasi-explicit parameters
        
        Parameters:
        ----------
        a, b, rho, m, sigma : float
            Raw SVI parameters
            
        Returns:
        -------
        tuple
            QE parameters (a, d, c, m, sigma)
        """
        c = b * sigma
        d = rho * c
        return a, d, c, m, sigma
    
    def calc_adc(self, iv, x, m, sigma):
        """
        Calculate optimal a, d, c parameters given fixed m and sigma
        
        Parameters:
        ----------
        iv : array-like
            Total variance values
        x : array-like
            Log-moneyness values
        m, sigma : float
            Fixed m and sigma parameters
            
        Returns:
        -------
        tuple
            QE parameters (a, d, c)
        """
        y = (x - m) / sigma
        s = max(sigma, 1e-6)
        
        # Upper bounds for parameters
        max_a = max(iv.max(), 1e-6)  # a should be positive
        max_dc = 2 * np.sqrt(2) * s  # Reasonable upper limits for d and c
        
        # Set bounds for nonnegative a, d, c
        bnd = ((0, 0, 0), (max_a, max_dc, max_dc))
        
        z = np.sqrt(y**2 + 1)
        
        # Coordinate rotation (45 degrees) for faster calculation
        A = np.column_stack([
            np.ones(len(iv)),
            np.sqrt(2)/2 * (y + z),
            np.sqrt(2)/2 * (-y + z)
        ])
        
        # Linear least squares with bounds
        result = opt.lsq_linear(A, iv, bnd, tol=1e-12, verbose=False)
        a_rot, d_rot, c_rot = result.x
        
        # Rotate back to original coordinates
        a = a_rot
        d = np.sqrt(2)/2 * (d_rot - c_rot)
        c = np.sqrt(2)/2 * (d_rot + c_rot)
        
        return a, d, c
    
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
        Check butterfly arbitrage constraint
        
        Parameters:
        ----------
        params : tuple
            SVI parameters (a, b, rho, m, sigma)
        k : array-like
            Log-moneyness k = log(K/F_T)
            
        Returns:
        -------
        float
            Minimum g value (should be > 0 for no arbitrage)
        """
        k_grid = np.linspace(np.min(k) - 0.5, np.max(k) + 0.5, 100)
        g_values = self.g_function(k_grid, params)
        return np.min(g_values)
    
    def calendar_spread_constraint(self, params_current, params_prev, k_current, k_prev=None):
        """
        Check calendar spread constraint
        
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
            Minimum difference (should be > 0 for no arbitrage)
        """
        if k_prev is None:
            k_prev = k_current
        
        # Create common grid
        k_min = min(np.min(k_current), np.min(k_prev))
        k_max = max(np.max(k_current), np.max(k_prev))
        grid_k = np.linspace(k_min - 0.5, k_max + 0.5, 100)
        
        # Evaluate variance functions
        w_current = self.svi_raw(grid_k, params_current)
        w_prev = self.svi_raw(grid_k, params_prev)
        
        # Return minimum difference
        return np.min(w_current - w_prev)
    
    def calibrate_single_qe(self, k, market_variance, initial_guess=None):
        """
        Calibrate SVI model for a single maturity using Quasi-Explicit method
        
        Parameters:
        ----------
        k : array-like
            Log-moneyness k = log(K/F_T)
        market_variance : array-like
            Market implied total variance
        initial_guess : tuple, optional
            Initial guess for SVI parameters (m, sigma)
            
        Returns:
        -------
        tuple
            Calibrated SVI parameters (a, b, rho, m, sigma)
        """
        k = np.asarray(k)
        market_variance = np.asarray(market_variance)
        
        # Initial guess for m and sigma if not provided
        if initial_guess is None:
            # m at minimum variance point
            min_var_idx = np.argmin(market_variance)
            init_m = k[min_var_idx]
            
            # sigma as a fraction of the range of k
            k_range = max(np.max(k) - np.min(k), 0.01)
            init_sigma = k_range / 4
            
            init_msigma = [init_m, init_sigma]
        else:
            # Extract m and sigma from initial parameters
            if len(initial_guess) == 5:
                # If raw SVI parameters provided
                _, _, _, init_m, init_sigma = initial_guess
            elif len(initial_guess) == 2:
                # If just m and sigma provided
                init_m, init_sigma = initial_guess
            else:
                raise ValueError("Initial guess should be either (m, sigma) or full SVI parameters")
            
            init_msigma = [init_m, init_sigma]
        
        # Two-step optimization
        opt_rmse = float('inf')
        
        # Define optimization function for m and sigma
        def opt_msigma(msigma):
            _m, _sigma = msigma
            _y = (k - _m) / _sigma 
            _a, _d, _c = self.calc_adc(market_variance, k, _m, _sigma)
            return np.sum(np.square(_a + _d * _y + _c * np.sqrt(_y**2 + 1) - market_variance))
        
        # Iterative two-step optimization
        for i in range(1, self.maxiter + 1):
            # Optimize m and sigma
            bounds = (
                (2*min(np.min(k), 0), 2*max(np.max(k), 0)),  # m bounds
                (1e-6, 1.0)  # sigma bounds
            )
            
            with SuppressOutput(suppress_stdout=not self.verbose):
                result = opt.minimize(
                    opt_msigma,
                    init_msigma,
                    method='Nelder-Mead',
                    bounds=bounds,
                    tol=1e-12
                )
            
            m_star, sigma_star = result.x
            
            # Calculate a, d, c based on optimized m, sigma
            a_star, d_star, c_star = self.calc_adc(market_variance, k, m_star, sigma_star)
            
            # Calculate RMSE
            y_star = (k - m_star) / sigma_star
            model_variance = a_star + d_star * y_star + c_star * np.sqrt(y_star**2 + 1)
            opt_rmse1 = np.sqrt(np.mean(np.square(model_variance - market_variance)))
            
            if self.verbose:
                print(f"round {i}: RMSE={opt_rmse1:.8f} para={[a_star, d_star, c_star, m_star, sigma_star]}")
            
            # Check convergence
            if i > 1 and opt_rmse - opt_rmse1 < self.exit_tol:
                break
            
            opt_rmse = opt_rmse1
            init_msigma = [m_star, sigma_star]
        
        # Convert QE parameters to raw SVI parameters
        raw_params = self.quasi2raw(a_star, d_star, c_star, m_star, sigma_star)
        
        if self.verbose:
            print(f"\nFinished. Raw SVI params = {np.round(raw_params, 10)}")
        
        return raw_params
    
    def calibrate_single(self, k, market_variance, initial_guess=None):
        """
        Calibrate SVI model for a single maturity - wrapper for QE method
        
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
        return self.calibrate_single_qe(k, market_variance, initial_guess)
    
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
        # First attempt with default initial guess
        try:
            params = self.calibrate_single(k, market_variance)
            
            # Check arbitrage constraints
            min_g = self.butterfly_arbitrage_constraint(params, k)
            
            if min_g >= -self.epsilon:
                # No butterfly arbitrage, good fit
                return params
            elif self.verbose:
                print(f"Warning: Butterfly arbitrage detected, min_g = {min_g}")
                
        except Exception as e:
            if self.verbose:
                print(f"First calibration attempt failed: {e}")
        
        # Try different initial guesses
        initial_guesses = [
            # m at different percentiles, sigma at different widths
            (np.median(k), (np.max(k) - np.min(k))/4),
            (np.percentile(k, 25), (np.max(k) - np.min(k))/3),
            (np.percentile(k, 75), (np.max(k) - np.min(k))/5),
        ]
        
        for i, guess in enumerate(initial_guesses[:max_attempts-1]):
            try:
                if self.verbose:
                    print(f"Attempting calibration with alternative initial guess {i+1}...")
                
                params = self.calibrate_single(k, market_variance, guess)
                
                # Check arbitrage constraints
                min_g = self.butterfly_arbitrage_constraint(params, k)
                
                if min_g >= -self.epsilon:
                    # No butterfly arbitrage, good fit
                    return params
                elif self.verbose:
                    print(f"Warning: Butterfly arbitrage detected, min_g = {min_g}")
                    
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
        sigma_simple = (np.max(k) - np.min(k)) / 4
        
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
            
            # First do regular calibration
            params = self.calibrate_single(
                k_list[i], 
                market_variance_list[i],
                initial_guess_list[i] or self.calibration_results[-1]
            )
            
            # Check calendar spread constraint
            prev_params = self.calibration_results[-1]
            min_diff = self.calendar_spread_constraint(
                params, prev_params, k_list[i], k_list[i-1]
            )
            
            if min_diff >= -self.epsilon:
                # No calendar spread arbitrage
                self.calibration_results.append(params)
            else:
                # Calendar spread arbitrage detected
                if self.verbose:
                    print(f"Warning: Calendar spread arbitrage detected, min_diff = {min_diff:.6f}")
                    print("Adjusting parameters to avoid calendar spread arbitrage...")
                
                # Simple adjustment: increase 'a' parameter to shift the curve up
                a, b, rho, m, sigma = params
                a_adjusted = a + abs(min_diff) + 2*self.epsilon
                adjusted_params = (a_adjusted, b, rho, m, sigma)
                
                # Verify adjustment worked
                min_diff_adjusted = self.calendar_spread_constraint(
                    adjusted_params, prev_params, k_list[i], k_list[i-1]
                )
                
                if self.verbose:
                    print(f"After adjustment, min_diff = {min_diff_adjusted:.6f}")
                
                self.calibration_results.append(adjusted_params)
        
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
            
            # First try with previous parameters as initial guess
            try:
                # Previous parameters as initial guess
                prev_params = self.calibration_results[-1]
                
                # Use previous m, sigma as initial guess for QE method
                _, _, _, prev_m, prev_sigma = prev_params
                initial_guess = (prev_m, prev_sigma)
                
                params = self.calibrate_single(
                    k_list[i], market_variance_list[i], initial_guess
                )
                
                # Check calendar spread constraint
                min_diff = self.calendar_spread_constraint(
                    params, prev_params, k_list[i], k_list[i-1]
                )
                
                if min_diff >= -self.epsilon:
                    # No calendar spread arbitrage
                    self.calibration_results.append(params)
                    continue
                else:
                    # Calendar spread arbitrage, try robust calibration
                    if self.verbose:
                        print(f"Warning: Calendar spread arbitrage detected, min_diff = {min_diff:.6f}")
                        print("Trying robust calibration...")
                        
                    raise ValueError("Calendar spread arbitrage detected")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Standard calibration failed: {e}")
                    print("Trying robust calibration...")
                
                # Try robust calibration
                params = self.robust_calibration(
                    k_list[i], market_variance_list[i], max_attempts
                )
                
                # Check calendar spread constraint
                prev_params = self.calibration_results[-1]
                min_diff = self.calendar_spread_constraint(
                    params, prev_params, k_list[i], k_list[i-1]
                )
                
                if min_diff < -self.epsilon:
                    if self.verbose:
                        print(f"Warning: Calendar spread arbitrage detected, min_diff = {min_diff:.6f}")
                        print("Adjusting parameters...")
                    
                    # Simple adjustment: increase 'a' parameter
                    a, b, rho, m, sigma = params
                    a_adjusted = a + abs(min_diff) + 2*self.epsilon
                    params = (a_adjusted, b, rho, m, sigma)
                    
                    # Verify adjustment worked
                    min_diff_adjusted = self.calendar_spread_constraint(
                        params, prev_params, k_list[i], k_list[i-1]
                    )
                    
                    if self.verbose:
                        print(f"After adjustment, min_diff = {min_diff_adjusted:.6f}")
                
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
