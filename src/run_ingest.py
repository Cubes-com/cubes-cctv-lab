import multiprocessing
import os
import sys
import time
import yaml
import signal
import subprocess

"""
Ingest Supervisor
-----------------
Reads `cameras.yml` and spawns an `ffmpeg` process for each camera.
It transcodes/remuxes the stream to a local MediaMTX instance for 
low-latency consumption by the analysis modules.
"""

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_ffmpeg_cmd(cam, defaults, mediamtx_host, mediamtx_port):
    name = cam["name"]
    url = cam["url"]

    fps = int(cam.get("fps", defaults.get("fps", 8)))
    width = int(cam.get("width", defaults.get("width", 960)))
    height = int(cam.get("height", defaults.get("height", 540)))

    rtsp_transport = cam.get("rtsp_transport", defaults.get("rtsp_transport", "tcp"))
    timeout_sec = int(cam.get("timeout_sec", defaults.get("timeout_sec", 5)))
    rw_timeout_us = timeout_sec * 1000000

    stereo_split = cam.get("stereo_split", False)
    
    gop = fps * 2
    
    
    # Use aggressive buffering for stereo_split cameras or if explicitly requested
    # These cameras tend to be high-res double-wide streams that need more buffer.
    use_aggressive_buffering = stereo_split or "2ndfloorworkshop" in name
    
    if use_aggressive_buffering:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
        ]
        # Aggressive buffering/probing
        cmd.extend([
            "-rtsp_transport", "tcp",
            "-analyzeduration", "10000000", # 10s (reduced from 20s to help startup/load)
            "-probesize", "10000000",       # 10MB (reduced from 20MB)
        ])
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
        ]
        # Standard low-latency flags for other cameras
        cmd.extend([
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-rtsp_transport", rtsp_transport,
            "-analyzeduration", "5000000",
            "-probesize", "5000000",
        ])
        
    cmd.extend(["-i", url])
    cmd.append("-an")
    
    # Common encoding params
    # Use ultrafast for split streams to save CPU, veryfast for others
    preset = "ultrafast" if stereo_split else "veryfast"
    
    enc_params = [
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-x264-params", "keyint=%d:min-keyint=%d:scenecut=0:repeat-headers=1" % (gop, gop),
        "-rtsp_transport", "tcp",
        "-f", "rtsp"
    ]
    
    if stereo_split:
        # Split width in half
        w2 = width // 2
        
        # Complex filter: crop left, crop right.
        filter_complex = (
            f"[0:v]crop=iw/2:ih:0:0,fps={fps},scale={w2}:{height}[left];"
            f"[0:v]crop=iw/2:ih:iw/2:0,fps={fps},scale={w2}:{height}[right]"
        )
        
        cmd.extend(["-filter_complex", filter_complex])
        
        # Map Left
        out_left = "rtsp://%s:%s/%s_left" % (mediamtx_host, mediamtx_port, name)
        cmd.extend(["-map", "[left]"])
        cmd.extend(enc_params)
        cmd.append(out_left)
        
        # Map Right
        out_right = "rtsp://%s:%s/%s_right" % (mediamtx_host, mediamtx_port, name)
        cmd.extend(["-map", "[right]"])
        cmd.extend(enc_params)
        cmd.append(out_right)
        
    else:
        vf = "fps=%d,scale=%d:%d" % (fps, width, height)
        out_url = "rtsp://%s:%s/%s" % (mediamtx_host, mediamtx_port, name)
        
        cmd.extend(["-vf", vf])
        cmd.extend(enc_params)
        cmd.append(out_url)

    return cmd
def spawn_loop(cam, defaults, mediamtx_host, mediamtx_port, logs_dir, stop_event):
    name = cam["name"]
    log_path = os.path.join(logs_dir, "%s.log" % name)

    while not stop_event.is_set():
        cmd = build_ffmpeg_cmd(cam, defaults, mediamtx_host, mediamtx_port)
        with open(log_path, "a") as logf:
            logf.write("\n--- starting ffmpeg for %s at %s ---\n" % (name, time.ctime()))
            logf.write("Cmd: %s\n" % " ".join(cmd))
            logf.flush()

            try:
                p = subprocess.Popen(cmd, stdout=logf, stderr=logf)
                while True:
                    if stop_event.is_set():
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        return

                    rc = p.poll()
                    if rc is not None:
                        logf.write("--- ffmpeg exited for %s with rc=%s at %s ---\n" % (name, rc, time.ctime()))
                        logf.flush()
                        
                        # Read remaining stderr
                        stderr_output = p.stderr.read()
                        if stderr_output:
                             logf.write(stderr_output)
                        break

                    time.sleep(0.5)

            except Exception as e:
                logf.write("--- failed to start ffmpeg for %s: %s ---\n" % (name, str(e)))
                logf.flush()

        # Backoff before restart
        time.sleep(1.0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_ingest.py cameras.yml")
        sys.exit(1)

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)

    mediamtx_host = os.environ.get("MEDIAMTX_HOST", cfg.get("mediamtx", {}).get("host", "127.0.0.1"))
    mediamtx_port = int(cfg.get("mediamtx", {}).get("rtsp_port", 8554))

    defaults = cfg.get("defaults", {})
    cameras = cfg.get("cameras", [])

    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    stop_event = multiprocessing.Event()

    def handle_sig(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    procs = []
    # One supervisor process per camera (simple + robust)
    for cam in cameras:
        # Using multiprocessing.Process to avoid GIL contention
        p = multiprocessing.Process(
            target=spawn_loop,
            args=(cam, defaults, mediamtx_host, mediamtx_port, logs_dir, stop_event),
            daemon=True
        )
        p.start()
        procs.append(p)

    print("Restreaming %d cameras into MediaMTX at rtsp://%s:%d/NAME" % (len(cameras), mediamtx_host, mediamtx_port))
    print("Press Ctrl+C to stop.")
    
    # Main loop just waits for stop signal
    while not stop_event.is_set():
        time.sleep(0.5)

    print("Stopping ingest processes...")
    for p in procs:
        p.terminate() # or join if we want them to finish cleanly, but terminate is faster for ffmpeg
        p.join()

if __name__ == "__main__":
    main()