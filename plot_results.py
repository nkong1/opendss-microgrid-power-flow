#!/usr/bin/env python3
"""
plot_results.py

Plotting utility for OpenDSS simulation results.
Can plot voltage profiles and feeder maps for any hour(s).

Usage:
    python plot_results.py --results results --hour 1501
    python plot_results.py --results results --hour 1501 --save
    python plot_results.py --results results --hour 150 200 500 --type voltage
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_results(results_dir):
    """Load simulation results from directory"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Load main results
    voltage_file = results_dir / "voltage_timeseries.csv"
    if not voltage_file.exists():
        raise FileNotFoundError(f"No voltage_timeseries.csv found in {results_dir}")
    
    df = pd.read_csv(voltage_file)
    
    # Load bus info (coordinates)
    bus_info_file = results_dir / "bus_info.csv"
    bus_info = None
    if bus_info_file.exists():
        bus_info = pd.read_csv(bus_info_file)
    
    # Load metadata
    metadata_file = results_dir / "simulation_metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return df, bus_info, metadata


def plot_voltage_vs_distance(df, hour, results_dir=None, save=False, highlight_pv=True):
    """
    Plot voltage magnitude vs distance from substation for a specific hour
    """
    # Filter for specific hour
    df_hour = df[df['hour'] == hour].copy()
    
    if df_hour.empty:
        raise ValueError(f"No data found for hour {hour}")
    
    # Remove buses without distance data
    df_hour = df_hour.dropna(subset=['distance_km'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each phase
    ax.scatter(df_hour['distance_km'], df_hour['pu_a'], 
               s=8, c='#e74c3c', label='Phase A', alpha=0.6, marker='o')
    ax.scatter(df_hour['distance_km'], df_hour['pu_b'], 
               s=8, c='#3498db', label='Phase B', alpha=0.6, marker='s')
    ax.scatter(df_hour['distance_km'], df_hour['pu_c'], 
               s=8, c='#2ecc71', label='Phase C', alpha=0.6, marker='^')
    
    # Highlight PV buses
    if highlight_pv and 'has_pv' in df_hour.columns:
        pv_buses = df_hour[df_hour['has_pv']]
        if not pv_buses.empty:
            ax.scatter(pv_buses['distance_km'], pv_buses['pu_a'],
                      s=150, c='gold', edgecolor='black', marker='*',
                      label='PV Buses', zorder=5, linewidths=1.5)
    
    # Add voltage limits
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Distance from Substation (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (per unit)', fontsize=12, fontweight='bold')
    ax.set_title(f'Voltage Profile - Hour {hour}', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # Set y-axis limits with some padding
    y_min = df_hour[['pu_a', 'pu_b', 'pu_c']].min().min()
    y_max = df_hour[['pu_a', 'pu_b', 'pu_c']].max().max()
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    
    # Save if requested
    if save and results_dir:
        output_file = Path(results_dir) / f"voltage_profile_hour_{hour:04d}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def plot_feeder_map(df, bus_info, hour=None, results_dir=None, save=False):
    """
    Plot feeder map with bus locations, optionally colored by voltage
    """
    if bus_info is None or bus_info.empty:
        raise ValueError("No bus coordinate information available for map plotting")
    
    # Remove buses without coordinates
    bus_info = bus_info.dropna(subset=['x', 'y', '']).copy()
    
    if bus_info.empty:
        raise ValueError("No buses have coordinate data")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if hour is not None:
        # Get voltage data for specific hour
        df_hour = df[df['hour'] == hour].copy()
        
        if df_hour.empty:
            raise ValueError(f"No data found for hour {hour}")
        
        # Calculate average voltage per bus
        df_hour['v_avg'] = df_hour[['pu_a', 'pu_b', 'pu_c']].mean(axis=1)
        
        # Merge with coordinates
        plot_data = bus_info.merge(df_hour[['bus', 'v_avg', 'has_pv']], on='bus', how='left')
        
        # Plot with color map
        scatter = ax.scatter(plot_data['x'], plot_data['y'], 
                           c=plot_data['v_avg'], s=40, 
                           cmap='RdYlGn', vmin=0.95, vmax=1.05,
                           edgecolors='black', linewidths=0.3, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Voltage (per unit)')
        cbar.ax.axhline(y=1.0, color='black', linewidth=1, linestyle='-')
        
        # Highlight PV buses
        if 'has_pv' in plot_data.columns:
            pv_buses = plot_data[plot_data['has_pv']]
            if not pv_buses.empty:
                ax.scatter(pv_buses['x'], pv_buses['y'],
                         s=200, marker='*', c='gold',
                         edgecolors='black', linewidths=1.5,
                         label='PV Buses', zorder=5)
        
        title = f'Feeder Map - Hour {hour}'
    else:
        # Plot without voltage coloring
        ax.scatter(bus_info['x'], bus_info['y'], 
                  s=20, c='steelblue', alpha=0.6,
                  edgecolors='black', linewidths=0.3)
        title = 'Feeder Map - All Buses'
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    if hour is not None and 'has_pv' in plot_data.columns:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save if requested
    if save and results_dir:
        hour_str = f"hour_{hour:04d}" if hour else "all"
        output_file = Path(results_dir) / f"feeder_map_{hour_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def plot_voltage_histogram(df, hour, results_dir=None, save=False):
    """Plot histogram of voltage magnitudes for a specific hour"""
    df_hour = df[df['hour'] == hour].copy()
    
    if df_hour.empty:
        raise ValueError(f"No data found for hour {hour}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    phases = [('pu_a', 'Phase A', '#e74c3c'), 
              ('pu_b', 'Phase B', '#3498db'), 
              ('pu_c', 'Phase C', '#2ecc71')]
    
    for ax, (col, label, color) in zip(axes, phases):
        data = df_hour[col].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Voltage (per unit)', fontweight='bold')
        ax.set_ylabel('Number of Buses', fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Voltage Distribution - Hour {hour}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save and results_dir:
        output_file = Path(results_dir) / f"voltage_histogram_hour_{hour:04d}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def plot_timeseries_stats(df, results_dir=None, save=False):
    """Plot voltage statistics over time"""
    # Calculate statistics per hour
    stats = df.groupby('hour').agg({
        'pu_a': ['mean', 'min', 'max'],
        'pu_b': ['mean', 'min', 'max'],
        'pu_c': ['mean', 'min', 'max']
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = stats['hour']
    
    # Plot mean values
    ax.plot(hours, stats[('pu_a', 'mean')], label='Phase A (mean)', color='#e74c3c', linewidth=1.5)
    ax.plot(hours, stats[('pu_b', 'mean')], label='Phase B (mean)', color='#3498db', linewidth=1.5)
    ax.plot(hours, stats[('pu_c', 'mean')], label='Phase C (mean)', color='#2ecc71', linewidth=1.5)
    
    # Fill between min and max
    ax.fill_between(hours, stats[('pu_a', 'min')], stats[('pu_a', 'max')], 
                    alpha=0.2, color='#e74c3c', label='Phase A (range)')
    ax.fill_between(hours, stats[('pu_b', 'min')], stats[('pu_b', 'max')], 
                    alpha=0.2, color='#3498db', label='Phase B (range)')
    ax.fill_between(hours, stats[('pu_c', 'min')], stats[('pu_c', 'max')], 
                    alpha=0.2, color='#2ecc71', label='Phase C (range)')
    
    # Add limits
    ax.axhline(y=1.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (per unit)', fontsize=12, fontweight='bold')
    ax.set_title('Voltage Statistics Over Time', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    if save and results_dir:
        output_file = Path(results_dir) / "voltage_timeseries_stats.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot OpenDSS simulation results')
    parser.add_argument('--results', required=True, help='Results directory')
    parser.add_argument('--hour', type=int, nargs='+', help='Hour(s) to plot')
    parser.add_argument('--type', choices=['voltage', 'map', 'histogram', 'timeseries', 'all'], 
                       default='all', help='Type of plot')
    parser.add_argument('--save', action='store_true', help='Save plots as PNG')
    parser.add_argument('--no-pv-highlight', action='store_true', help='Do not highlight PV buses')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    df, bus_info, metadata = load_results(args.results)
    print(f"  Loaded {len(df)} records")
    
    if metadata:
        print(f"  Simulation: hours {metadata['start_hour']} to {metadata['end_hour']}")
        print(f"  Total buses: {metadata['total_buses']}, PV buses: {metadata['pv_buses']}")
    
    highlight_pv = not args.no_pv_highlight
    
    # If no hour specified, show timeseries stats
    if args.hour is None:
        if args.type in ['all', 'timeseries']:
            plot_timeseries_stats(df, args.results, args.save)
        else:
            print("Error: --hour required for voltage, map, and histogram plots")
        return
    
    # Plot for each specified hour
    for hour in args.hour:
        print(f"\nPlotting hour {hour}...")
        
        if args.type in ['voltage', 'all']:
            plot_voltage_vs_distance(df, hour, args.results, args.save, highlight_pv)
        
        if args.type in ['map', 'all'] and bus_info is not None:
            plot_feeder_map(df, bus_info, hour, args.results, args.save)
        
        if args.type in ['histogram', 'all']:
            plot_voltage_histogram(df, hour, args.results, args.save)


if __name__ == "__main__":
    main()