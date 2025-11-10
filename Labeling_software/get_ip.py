#!/usr/bin/env python3
"""
Quick script to get your local IP address for sharing the labeling tool.
"""
import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        # (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "Unable to determine IP"

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"Your local IP address: {ip}")
    print(f"{'='*50}")
    print(f"\nShare this URL with your team:")
    print(f"  http://{ip}:5001")
    print(f"\nMake sure the server is running with:")
    print(f"  python app.py")
    print(f"{'='*50}\n")

