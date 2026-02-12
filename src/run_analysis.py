import os
import sys
import time
import yaml
import signal
import subprocess

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_analysis.py cameras.yml [optional_filter_name]")
        sys.exit(1)

    cfg_path = sys.argv[1]
    filter_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    cfg = load_config(cfg_path)

    # Local MediaMTX where ingest is publishing
    mediamtx_host = os.environ.get("MEDIAMTX_HOST", "127.0.0.1")
    mediamtx_port = 8554
    cameras = cfg.get("cameras", [])

    procs = []
    
    stop_flag = {"stop": False}
    def handle_sig(signum, frame):
        print("\nStopping analysis manager...")
        stop_flag["stop"] = True
        for p in procs:
            p.terminate()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    print(f"Starting analysis manager for {len(cameras)} cameras...")

    for cam in cameras:
        name = cam["name"]
        
        # Filter support
        if filter_name and filter_name != "all" and name != filter_name:
            continue

        stereo_split = cam.get("stereo_split", False)
        
        targets = []
        if stereo_split:
            targets.append((f"{name}_left", f"rtsp://{mediamtx_host}:{mediamtx_port}/{name}_left"))
            targets.append((f"{name}_right", f"rtsp://{mediamtx_host}:{mediamtx_port}/{name}_right"))
        else:
            targets.append((name, f"rtsp://{mediamtx_host}:{mediamtx_port}/{name}"))

        for t_name, t_url in targets:
            cmd = [
                sys.executable, 
                "src/analyse_rtsp.py",
                "--name", t_name,
                "--url", t_url
            ]
            
            print(f"Launching analysis for {t_name} -> {t_url}")
            p = subprocess.Popen(cmd)
            procs.append(p)
            
            # Stagger startup
            time.sleep(2)

    while not stop_flag["stop"]:
        # Check if any process died
        for p in procs:
            if p.poll() is not None:
                print(f"Process {p.pid} exited.")
                procs.remove(p)
        
        if not procs:
            print("All analysis processes exited.")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()
