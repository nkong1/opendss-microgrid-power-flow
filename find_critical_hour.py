#!/usr/bin/env python3
"""
find_critical_hour.py

Finds the hour with the highest net generation (PV - Load).
This hour is most critical for voltage analysis as it represents peak reverse power flow.

Usage:
    python find_critical_hour.py --pv_file pv_data.xlsx --load_file dss_files\load_timeseries_data.csv
    python find_critical_hour.py --pv_file pv_data.xlsx --load_file dss_files\load_timeseries_data.csv --top 10
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_pv_excel(pv_excel_path):
    """Load PV data from Excel file"""
    p = Path(pv_excel_path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {pv_excel_path}")

    xls = pd.ExcelFile(p)
    df_list = []

    for sheet in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sheet)
        required = ["hour_index", "bus_id", "pv_kW"]
        for col in required:
            if col not in tmp.columns:
                raise ValueError(f"Sheet '{sheet}' missing column '{col}'")
        
        tmp = tmp[required].copy()
        tmp["hour_index"] = tmp["hour_index"].astype(int)
        tmp["pv_kW"] = tmp["pv_kW"].astype(float).fillna(0) / 1000 # data is actually given in watts
        df_list.append(tmp)

    combined = pd.concat(df_list, ignore_index=True)
    return combined


def load_load_profiles(load_profile_path):
    """Load load profiles and convert to hourly"""
    p = Path(load_profile_path)
    if not p.exists():
        raise FileNotFoundError(f"Load profile file not found: {load_profile_path}")
    
    df = pd.read_csv(p)
    
    time_col = df.columns[0]
    df = df.rename(columns={time_col: 'time_index'})
    df['hour_index'] = df['time_index'] // 4 + 1
    
    bus_columns = [col for col in df.columns if col not in ['time_index', 'hour_index']]
    hourly_loads = df.groupby('hour_index')[bus_columns].mean().reset_index()
    
    load_list = []
    for bus in bus_columns:
        bus_data = hourly_loads[['hour_index', bus]].copy()
        bus_data.columns = ['hour_index', 'load_kW']
        bus_data['bus_id'] = bus
        load_list.append(bus_data)
    
    combined = pd.concat(load_list, ignore_index=True)
    combined = combined[['hour_index', 'bus_id', 'load_kW']]
    combined['load_kW'] = combined['load_kW'].fillna(0.0)
    
    return combined


def find_critical_hours(pv_df, load_df, top_n=8760, show_plot=False):
    """Find hours with highest net generation (PV - Load)"""
    
    # Calculate total PV per hour
    pv_hourly = pv_df.groupby('hour_index')['pv_kW'].sum().reset_index()
    pv_hourly.columns = ['hour_index', 'total_pv_kW']
    
    # Calculate total load per hour
    load_hourly = load_df.groupby('hour_index')['load_kW'].sum().reset_index()
    load_hourly.columns = ['hour_index', 'total_load_kW']
    
    # Merge and calculate net generation
    net_gen = pv_hourly.merge(load_hourly, on='hour_index', how='outer').fillna(0)
    net_gen['net_generation_kW'] = net_gen['total_pv_kW'] - net_gen['total_load_kW']
    net_gen['pv_to_load_ratio'] = net_gen['total_pv_kW'] / (net_gen['total_load_kW'] + 1e-6)
    
    # Sort by net generation (descending)
    net_gen_sorted = net_gen.sort_values('net_generation_kW', ascending=False).reset_index(drop=True)
    
    # Get top N hours
    top_hours = net_gen_sorted.head(top_n)
    
    print("\n" + "="*80)
    print("CRITICAL HOURS ANALYSIS")
    print("="*80)
    print(f"\nTop {top_n} hours with highest net generation (PV - Load):\n")
    print(f"{'Rank':<6} {'Hour':<8} {'Total PV (kW)':<15} {'Total Load (kW)':<17} {'Net Gen (kW)':<15} {'PV/Load Ratio':<15}")
    print("-" * 80)
    
    for idx, row in top_hours.iterrows():
        print(f"{idx+1:<6} {int(row['hour_index']):<8} {row['total_pv_kW']:>13,.1f}  {row['total_load_kW']:>15,.1f}  {row['net_generation_kW']:>13,.1f}  {row['pv_to_load_ratio']:>13.2f}")
    
    print("\n" + "="*80)
    print(f"\n⚠️  MOST CRITICAL HOUR: {int(top_hours.iloc[0]['hour_index'])}")
    print(f"    This hour has the highest reverse power flow and is most likely")
    print(f"    to cause overvoltage issues from PV generation.")
    print("="*80 + "\n")
    
    # Show some statistics
    print("Summary Statistics:")
    print(f"  Total hours analyzed: {len(net_gen)}")
    print(f"  Hours with net generation (PV > Load): {(net_gen['net_generation_kW'] > 0).sum()}")
    print(f"  Hours with net consumption (Load > PV): {(net_gen['net_generation_kW'] < 0).sum()}")
    print(f"  Max net generation: {net_gen['net_generation_kW'].max():,.1f} kW (Hour {int(net_gen.loc[net_gen['net_generation_kW'].idxmax(), 'hour_index'])})")
    print(f"  Max net consumption: {net_gen['net_generation_kW'].min():,.1f} kW (Hour {int(net_gen.loc[net_gen['net_generation_kW'].idxmin(), 'hour_index'])})")
    
    # Optional plot
    if show_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: PV vs Load over time
        ax1 = axes[0]
        ax1.plot(net_gen['hour_index'], net_gen['total_pv_kW'], 
                label='Total PV Generation', color='orange', linewidth=1.5, alpha=0.8)
        ax1.plot(net_gen['hour_index'], net_gen['total_load_kW'], 
                label='Total Load', color='blue', linewidth=1.5, alpha=0.8)
        
        # Highlight critical hour
        critical_hour = top_hours.iloc[0]['hour_index']
        ax1.axvline(x=critical_hour, color='red', linestyle='--', linewidth=2, 
                   label=f'Most Critical Hour ({int(critical_hour)})', alpha=0.7)
        
        ax1.set_xlabel('Hour Index', fontweight='bold')
        ax1.set_ylabel('Power (kW)', fontweight='bold')
        ax1.set_title('PV Generation vs Load Over Time', fontweight='bold', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Net generation
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'gray' for x in net_gen['net_generation_kW']]
        ax2.bar(net_gen['hour_index'], net_gen['net_generation_kW'], 
               color=colors, alpha=0.6, width=1.0, edgecolor='none')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.axvline(x=critical_hour, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Hour Index', fontweight='bold')
        ax2.set_ylabel('Net Generation (kW)', fontweight='bold')
        ax2.set_title('Net Generation (PV - Load) - Green = Excess PV', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('critical_hours_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: critical_hours_analysis.png")
        plt.show()
    
    return top_hours


def main():
    parser = argparse.ArgumentParser(
        description='Find critical hours with highest net PV generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_critical_hour.py --pv_file pv_data.xlsx --load_file loads.xlsx
  python find_critical_hour.py --pv_file pv_data.xlsx --load_file loads.xlsx --top 20 --plot
        """
    )
    parser.add_argument('--pv_file', required=True, help='Path to PV .xlsx file')
    parser.add_argument('--load_file', required=True, help='Path to load profile file')
    parser.add_argument('--top', type=int, default=8760, help='Number of top hours to display (default: 10)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plot')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading PV data...")
    pv_df = load_pv_excel(args.pv_file)
    print(f"  ✓ Loaded {len(pv_df)} PV records")
    
    print("Loading load profiles...")
    load_df = load_load_profiles(args.load_file)
    print(f"  ✓ Loaded {len(load_df)} load records")
    
    # Find critical hours
    critical_hours = find_critical_hours(pv_df, load_df, top_n=args.top, show_plot=args.plot)
    
    # Save to CSV
    critical_hours.to_csv('critical_hours.csv', index=False)
    print(f"\n✓ Critical hours saved to: critical_hours.csv")


if __name__ == "__main__":
    main()