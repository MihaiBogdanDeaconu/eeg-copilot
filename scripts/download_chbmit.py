import os

def main():
    """
    Provides instructions for downloading the CHB-MIT Scalp EEG Database.
    """
    print("=====================================================================")
    print("AI Co-Pilot: CHB-MIT Scalp EEG Database Download Instructions")
    print("=====================================================================")
    print("The CHB-MIT database is hosted on PhysioNet, a repository of medical research data.")
    print("\n--- Step-by-Step Instructions ---")
    print("1. Go to the PhysioNet project page: https://physionet.org/content/chbmit/1.0.0/")
    print("2. You will need a PhysioNet account to download the data. If you don't have one, register on the PhysioNet website.")
    print("3. Once logged in, you can download the data. You can download the entire dataset as a zip file or use tools like 'wget' to download individual files.")
    print("   The total size is approximately 25 GB.")
    print("4. After downloading, unzip the files into a directory.")
    print("\n--- Final Step ---")
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'CHB-MIT')
    print(f"5. Place the entire dataset into the following directory: {data_dir}")
    print("   (You may need to create the 'data/CHB-MIT' directories).")
    print("6. Finally, update the `paths.chb_mit_dir` key in `config/config.yaml` to point to this directory.")
    print("\nThis script does not perform the download automatically due to the access restrictions.")
    print("=====================================================================")

if __name__ == "__main__":
    main()