{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "47f14e7d",
   "metadata": {},
   "source": [
    "# Radial Velocity Analysis of 55 Cancri System\n",
    "\n",
    "This notebook fits a 2-planet model to the HD 75732 (55 Cancri) radial velocity data and estimates the planetary masses."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca4ee1e5",
   "metadata": {},
   "source": [
    "## Step 1: Load the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d83cb81",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.optimize import curve_fit\n",
    "\n",
    "# Load the dataset (assuming you downloaded and saved it as '55Cnc_rvdata.txt')\n",
    "data = np.loadtxt(\"55Cnc_rvdata.txt\")\n",
    "\n",
    "# Columns: time (HJD), RV (m/s), uncertainty (m/s)\n",
    "time = data[:, 0]\n",
    "rv = data[:, 1]\n",
    "rv_err = data[:, 2]\n",
    "\n",
    "# Time range for smoother plotting\n",
    "t_fit = np.linspace(min(time), max(time), 1000)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b351199",
   "metadata": {},
   "source": [
    "## Step 2: Define 2-Planet Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4936dbda",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rv_model_two_planets(t, K1, P1, phi1, K2, P2, phi2, gamma):\n",
    "    return (\n",
    "        K1 * np.sin(2 * np.pi * t / P1 + phi1) +\n",
    "        K2 * np.sin(2 * np.pi * t / P2 + phi2) +\n",
    "        gamma\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ad95879",
   "metadata": {},
   "source": [
    "## Step 3: Fit the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af9f08d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initial guesses for parameters: [K1, P1, phi1, K2, P2, phi2, gamma]\n",
    "initial_guess_2p = [100, 14.6, 0, 50, 44.3, 0, 0]\n",
    "\n",
    "# Curve fitting\n",
    "popt_2p, pcov_2p = curve_fit(rv_model_two_planets, time, rv, sigma=rv_err, p0=initial_guess_2p, absolute_sigma=True)\n",
    "\n",
    "# Extract parameters\n",
    "K1_fit, P1_fit, phi1_fit, K2_fit, P2_fit, phi2_fit, gamma_fit_2p = popt_2p\n",
    "\n",
    "# Evaluate the fitted model\n",
    "rv_fit_2p = rv_model_two_planets(t_fit, *popt_2p)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6935825",
   "metadata": {},
   "source": [
    "## Step 4: Plot the Fit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13ef45e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 6))\n",
    "plt.errorbar(time, rv, yerr=rv_err, fmt='o', markersize=4, label='Observed RV', ecolor='gray')\n",
    "plt.plot(t_fit, rv_fit_2p, 'r-', label='2-Planet Fit', linewidth=2)\n",
    "plt.xlabel(\"Time [HJD]\")\n",
    "plt.ylabel(\"Radial Velocity [m/s]\")\n",
    "plt.title(\"2-Planet Model Fit to HD 75732 RV Data\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"hd75732_2planet_fit.pdf\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3362d9fd",
   "metadata": {},
   "source": [
    "## Step 5: Residual Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07e11586",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate residuals\n",
    "rv_model_vals_2p = rv_model_two_planets(np.array(time), *popt_2p)\n",
    "residuals_2p = np.array(rv) - rv_model_vals_2p\n",
    "\n",
    "# Plot residuals over time\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.errorbar(time, residuals_2p, yerr=rv_err, fmt='o', markersize=4)\n",
    "plt.axhline(0, color='red', linestyle='--')\n",
    "plt.xlabel(\"Time [HJD]\")\n",
    "plt.ylabel(\"Residuals [m/s]\")\n",
    "plt.title(\"Residuals of 2-Planet Model\")\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"hd75732_2planet_residuals.pdf\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8e7fdee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Histogram of residuals\n",
    "residual_std_2p = np.std(residuals_2p)\n",
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "q25, q75 = np.percentile(residuals_2p, [25, 75])\n",
    "binwidth = 2 * (q75 - q25) * len(residuals_2p) ** (-1/3)\n",
    "bins = int((max(residuals_2p) - min(residuals_2p)) / binwidth)\n",
    "plt.hist(residuals_2p, bins=bins, color='lightcoral', edgecolor='black')\n",
    "plt.axvline(np.mean(residuals_2p), color='red', linestyle='dashed', label=f'Mean = {np.mean(residuals_2p):.2f}')\n",
    "plt.axvline(np.mean(residuals_2p) + residual_std_2p, color='green', linestyle='dashed', label=f'+1σ = {residual_std_2p:.2f}')\n",
    "plt.axvline(np.mean(residuals_2p) - residual_std_2p, color='green', linestyle='dashed', label=f'-1σ = {residual_std_2p:.2f}')\n",
    "plt.xlabel(\"Residual [m/s]\")\n",
    "plt.ylabel(\"Number of Observations\")\n",
    "plt.title(\"Histogram of 2-Planet Model Residuals\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"hd75732_2planet_residuals_histogram.pdf\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c66021f",
   "metadata": {},
   "source": [
    "## Step 6: Estimate Planetary Masses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4a3fcc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stellar mass in solar masses\n",
    "M_star = 0.905\n",
    "\n",
    "def calc_planet_mass(K, P, M_star):\n",
    "    return (K / 28.4) * (P / 365) ** (1/3) * M_star ** (2/3)\n",
    "\n",
    "mass1 = calc_planet_mass(K1_fit, P1_fit, M_star)\n",
    "mass2 = calc_planet_mass(K2_fit, P2_fit, M_star)\n",
    "\n",
    "print(f\"Planet 1: K = {K1_fit:.2f} m/s, P = {P1_fit:.2f} days --> M = {mass1:.3f} M_jup\")\n",
    "print(f\"Planet 2: K = {K2_fit:.2f} m/s, P = {P2_fit:.2f} days --> M = {mass2:.3f} M_jup\")"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
