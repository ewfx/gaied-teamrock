import subprocess

def main():
    # Run the Streamlit UI script
    subprocess.run(["streamlit", "run", "app/StreamlitUI.py"])

if __name__ == "__main__":
    main()