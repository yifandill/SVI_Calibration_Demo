import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
from matplotlib.gridspec import GridSpec


def plot_forward_curve(dir: str, date: str, spot: float, df_fwd: pd.DataFrame):
    """
    Plot the forward curve for a given date and spot price.
    """
    # Create a new dataframe including the spot price
    df_plot = pd.DataFrame(index=[pd.Timestamp(date)] + list(df_fwd.index), data={'forward': [spot] + list(df_fwd['forward'])})

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot['forward'], marker='o', linestyle='-', linewidth=2)
    plt.scatter(df_plot.index[0], spot, color='red', s=100, zorder=5, label='Spot Price')
    plt.scatter(df_plot.index[1:], df_fwd['forward'], color='blue', s=50, label='Forward Prices')

    # Format the plot
    plt.title(f'AAPL Forward Curve, {date}', fontsize=15)
    plt.xlabel('Expiry Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Format x-axis to show dates better
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(f"{dir}/forward_curve_{date}.png", dpi=300)
    plt.close()


def plot_raw_smile(dir: str, date: str, expiry: str, K: np.ndarray, iv: np.ndarray):
    """
    Plot the implied volatility smile for a given date and expiry, i.e., plot the implied volatility against the strikes.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(K, iv, marker='o', s=50, alpha=0.8)
    plt.title(f'AAPL Implied Volatility Smile, {date}, expiry {expiry}', fontsize=15)
    plt.xlabel('Strikes', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{dir}/smile_{date}_ex_{expiry}.png", dpi=300)
    plt.close()


def plot_single_fit(calibrator, k, market_variance, params=None, title="SVI Calibration"):
    """
    Plot the SVI model fit against market data of a slice, i.e., one maturity.
    
    Parameters:
    calibrator : SVICalibrator - The calibrator instance for SVI calculations
    k : array-like - Log-moneyness k = log(K/F_T)
    market_variance : array-like - Market implied variances
    params : tuple - SVI parameters (a, b, rho, m, sigma)
    title : str - Plot title
    """
    if params is None and len(calibrator.calibration_results) > 0:
        params = calibrator.calibration_results[-1]
    elif params is None:
        raise ValueError("No calibration results available and no parameters provided")
        
    # Generate smooth curve for SVI model
    k_min, k_max = np.min(k) - 0.2, np.max(k) + 0.2
    k_smooth = np.linspace(k_min, k_max, 1000)
    model_variance = calibrator.svi_raw(k_smooth, params)
    
    plt.figure(figsize=(10, 6))
    
    # Add vertical line at k=0 (at-the-money)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot market data and model fit with matching colors
    plt.scatter(k, market_variance, label='Market Data', color='blue', alpha=0.8)
    plt.plot(k_smooth, model_variance, label='SVI Model', color='red', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Log moneyness $k$', fontsize=12)
    plt.ylabel('Implied Total Variance', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Format tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print calibrated parameters
    a, b, rho, m, sigma = params
    print(f"Calibrated parameters:")
    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print(f"rho = {rho:.6f}")
    print(f"m = {m:.6f}")
    print(f"sigma = {sigma:.6f}")
    
    # Check for arbitrage
    g_values = calibrator.g_function(k_smooth, params)
    min_g = np.min(g_values)
    print(f"Minimum g value: {min_g:.6f} (should be > 0 for no butterfly arbitrage)")


def plot_multi_fit(dir: str, date: str, calibrator, k_list, market_variance_list, params_list=None, maturities=None, show=False):
    """
    Plot the SVI model fit against market data of a surface, i.e., multiple maturities.
    
    Parameters:
    dir : str - Directory to save the plots
    date : str - Date for the plot title
    calibrator : SVICalibrator - The calibrator instance for SVI calculations
    k_list : list of arrays - Log-moneyness for each maturity (can be different lengths/values)
    market_variance_list : list of arrays - Market implied variances for each maturity
    params_list : list of tuples - Calibrated parameters for each maturity
    maturities : list of float - Time to maturity in years
    show : bool - If True, display the plot instead of saving it
    """
    if params_list is None:
        if not calibrator.calibration_results:
            raise ValueError("No calibration results available and no parameters provided")
        params_list = calibrator.calibration_results
        
    n_maturities = len(k_list)
    if maturities is None:
        maturities = [f"{i+1}/12" for i in range(n_maturities)]
    
    # Find global min and max across all k values for consistent plot limits
    k_min = min(np.min(k) for k in k_list)
    k_max = max(np.max(k) for k in k_list)
    k_range = [k_min - 0.2, k_max + 0.2]
    
    plt.figure(figsize=(10, 7))
    
    # Add vertical line at k=0 (at-the-money)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    
    # Create a custom colormap from purple to red (similar to image)
    colors = ['purple', 'blue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
    n_colors = max(n_maturities, len(colors))
    custom_cmap = LinearSegmentedColormap.from_list('maturity_cmap', colors, N=n_colors)
    plot_colors = [custom_cmap(i/max(1, n_maturities-1)) for i in range(n_maturities)]
    
    # Plot all maturities
    for i in range(n_maturities):
        # Format maturity label with 8 decimal places as requested
        if isinstance(maturities[i], float):
            maturity_label = f"T={maturities[i]:.8f}"
        else:
            maturity_label = f"T={maturities[i]}"
        
        # Market data points
        plt.scatter(
            k_list[i], market_variance_list[i], 
            color=plot_colors[i], alpha=0.8, marker='o', s=30
            )
        
        # Model fit - use a smooth curve with same color
        k_smooth = np.linspace(k_range[0], k_range[1], 1000)
        model_variance = calibrator.svi_raw(k_smooth, params_list[i])
        plt.plot(
            k_smooth, model_variance, 
            label=maturity_label, 
            color=plot_colors[i], linewidth=2
        )
    
    plt.title(f'SVI Calibration - Multiple Maturities, {date}', fontsize=14, fontweight='bold')
    plt.xlabel('Log moneyness $k$', fontsize=12)
    plt.ylabel('Implied Total Variance', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{dir}/multi_fit_{date}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_svi_surface(dir: str, date: str, calibrator, params_list, maturities=None, k_range=None, show=False):
    """
    Plot a 3D surface of the SVI model across multiple maturities.
    
    Parameters:
    dir : str - Directory to save the plots
    date : str - Date for the plot title
    calibrator : SVICalibrator - The calibrator instance for SVI calculations
    params_list : list of tuples - Calibrated parameters for each maturity
    maturities : list of float - Time to maturity in years
    k_range : tuple - Range of log-moneyness for x-axis (min, max)
    show : bool - If True, display the plot instead of saving it
    """
    if maturities is None:
        maturities = np.linspace(0.1, 1.0, len(params_list))
    
    # Create grid for surface
    if k_range is None:
        k_range = (-1.0, 1.0)
    
    k_range_values = np.linspace(k_range[0], k_range[1], 100)
    t_range = np.array(maturities)
    K, T = np.meshgrid(k_range_values, t_range)
    
    # Calculate variance surface
    Z = np.zeros_like(K)
    for i, params in enumerate(params_list):
        Z[i, :] = calibrator.svi_raw(k_range_values, params)
    
    # Convert variance to volatility
    Z = np.sqrt(Z)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a colormap similar to the 2D plot
    custom_cmap = plt.cm.viridis
    
    # Plot surface
    surf = ax.plot_surface(
        K, T, Z, cmap=custom_cmap, alpha=0.8, 
        linewidth=0, antialiased=True
    )
    
    # Plot individual smiles as lines on the surface
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))
    for i, t in enumerate(maturities):
        k_smooth = np.linspace(k_range[0], k_range[1], 100)
        variance = calibrator.svi_raw(k_smooth, params_list[i])
        vols = np.sqrt(variance)
        ax.plot(k_smooth, [t] * len(k_smooth), vols, color='red', linewidth=2)
    
    # Add a vertical plane at k=0 (at-the-money)
    k0_points = np.zeros(100)
    t_points = np.linspace(min(maturities), max(maturities), 100)
    vols_at_k0 = np.zeros(100)
    for i, t in enumerate(t_points):
        # Interpolate maturity
        if t <= min(maturities):
            params = params_list[0]
        elif t >= max(maturities):
            params = params_list[-1]
        else:
            # Find closest maturities and interpolate
            idx = np.searchsorted(maturities, t)
            t1, t2 = maturities[idx-1], maturities[idx]
            w = (t - t1) / (t2 - t1)
            params1, params2 = params_list[idx-1], params_list[idx]
            # Simple linear interpolation of variance
            var1 = calibrator.svi_raw(0, params1)
            var2 = calibrator.svi_raw(0, params2)
            var_interp = var1 * (1-w) + var2 * w
            vols_at_k0[i] = np.sqrt(var_interp)
    
    ax.plot(k0_points, t_points, vols_at_k0, color='black', linewidth=2)
    
    ax.set_xlabel('Log moneyness $k$', fontsize=12)
    ax.set_ylabel('Maturity $T$', fontsize=12)
    ax.set_zlabel('Implied Volatility', fontsize=12)
    ax.set_title(f"SVI Volatility Surface, {date}", fontsize=14, fontweight='bold')
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f"{dir}/svi_surface_{date}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_all_residuals(dir: str, date: str, evaluator, k_list, market_variance_list, params_list, maturities, subplots_per_row=3):
    """
    Plot SVI model fit residuals for all maturities in a grid layout with residual panels below each plot.
    
    Parameters:
    -----------
    dir : str
        Directory to save the plots
    date : str
        Date for the plot title
    evaluator : SVICalibrationEvaluator
        The calibration evaluator instance
    k_list : list of arrays
        Log-moneyness values for each maturity
    market_variance_list : list of arrays
        Market implied variances for each maturity
    params_list : list of tuples
        Calibrated SVI parameters for each maturity
    maturities : list of float
        The maturity values
    subplots_per_row : int
        Number of subplots to display in each row
    
    Returns:
    --------
    dict
        Dictionary of residual statistics for each maturity
    """
    n_maturities = len(k_list)
    
    # Calculate the number of rows needed
    n_rows = math.ceil(n_maturities / subplots_per_row)
    
    # Create a figure with a more square-like shape
    fig = plt.figure(figsize=(15, 15 * n_rows / subplots_per_row))
    
    # Store residual statistics
    all_residual_stats = {}
    
    # Create GridSpec for better control over subplot layout
    gs = GridSpec(n_rows*4, subplots_per_row, figure=fig)
    
    for i in range(n_maturities):
        k = k_list[i]
        market_variance = market_variance_list[i]
        params = params_list[i]
        maturity = maturities[i]
        
        # Calculate residuals
        model_variance = evaluator.calibrator.svi_raw(k, params)
        residuals = model_variance - market_variance
        
        # Calculate row and column indices
        row = i // subplots_per_row
        col = i % subplots_per_row
        
        # Create main subplot (3/4 of the height)
        ax_main = fig.add_subplot(gs[4*row:4*row+3, col])
        
        # Separate residual subplot (1/4 of the height)
        ax_resid = fig.add_subplot(gs[4*row+3, col])
        
        # Plot market data and model fit in main plot
        ax_main.scatter(k, market_variance, color='blue', marker='o', label='Market')
        
        # Generate smooth curve for model
        k_smooth = np.linspace(min(k) - 0.1, max(k) + 0.1, 100)
        model_smooth = evaluator.calibrator.svi_raw(k_smooth, params)
        ax_main.plot(k_smooth, model_smooth, color='red', linewidth=2, label='SVI')
        
        # Calculate statistics and add to plot
        rmse = np.sqrt(np.mean(residuals**2))
        min_residual = np.min(residuals)
        max_residual = np.max(residuals)
        
        stats_text = f"RMSE: {rmse:.6f}\nMin: {min_residual:.6f}\nMax: {max_residual:.6f}"
        ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', alpha=0.1))
        
        # Store stats
        stats = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_residual': max_residual,
            'min_residual': min_residual,
            'rmse': rmse
        }
        all_residual_stats[f"{maturity:.8f}"] = stats
        
        # Plot residuals in smaller subplot
        ax_resid.bar(k, residuals, width=0.02, color='#6666aa', alpha=0.8)  # Purple-blue color similar to image
        ax_resid.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Set labels for the residual plot
        ax_resid.set_xlabel('k', fontsize=8)
        ax_resid.set_ylabel('Resid.', fontsize=8, rotation=0, labelpad=15, va='center')
        
        # Set titles
        ax_main.set_title(f"T={maturity:.8f}", fontsize=10)
        
        # Only add legend to first subplot
        if i == 0:
            ax_main.legend(fontsize=8)
            
        # Remove xticks from main plot (residual plot has them)
        ax_main.set_xticklabels([])
        
        # Format residual plot y-limits to be symmetrical if needed
        max_abs_resid = max(abs(np.min(residuals)), abs(np.max(residuals)))
        y_margin = 0.0005  # Small margin beyond the data
        ax_resid.set_ylim(-max_abs_resid-y_margin, max_abs_resid+y_margin)
        
        # Set fixed y-limit for main plot based on data range
        y_max = max(np.max(market_variance), np.max(model_smooth)) * 1.1
        ax_main.set_ylim(0, y_max)
        
        # Match x-axis limits between main and residual plots
        x_min, x_max = min(k) - 0.1, max(k) + 0.1
        ax_main.set_xlim(x_min, x_max)
        ax_resid.set_xlim(x_min, x_max)
        
        # Format ticks
        ax_main.tick_params(axis='both', which='major', labelsize=8)
        ax_resid.tick_params(axis='both', which='major', labelsize=8)
        
    # Add an overall title
    fig.suptitle(f"RESIDUAL ANALYSIS (ALL MATURITIES), {date}", fontsize=14, y=0.98)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05, wspace=0.25, top=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.savefig(f"{dir}/residuals_{date}.png", dpi=300)
    
    return all_residual_stats
