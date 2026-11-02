import os
import argparse
from synaptic_dag.diagnostics import plot_publication_dashboard

def main():
    parser = argparse.ArgumentParser(description="Generate 300 DPI Neural Diagnostics Visuals")
    parser.add_argument("--output", type=str, default="diagnostics_dashboard_300dpi.png")
    args = parser.parse_args()
    
    print(f"[*] Rendering 300 DPI master dashboard to {args.output}...")
    plot_publication_dashboard(args.output)
    print("[+] Done!")

if __name__ == "__main__":
    main()
